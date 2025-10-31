import json
import camelot
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from scraping import sha

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")

from pathlib import Path

def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []
    main_html = str(main_el)
    tmp_path = file_path.with_suffix(".main.html")
    tmp_path.write_text(main_html, encoding="utf-8")

    elements = partition_html(filename=str(tmp_path), include_page_breaks=False, languages=["ita", "eng"])
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    if not elements:
        print(f"[WARN] partition_html non ha trovato elementi in {source_url}, uso fallback BeautifulSoup")
        text_fallback = main_el.get_text(separator="\n", strip=True)
        if text_fallback:
            docs.append(Document(
                page_content=text_fallback,
                metadata={
                    "source_url": source_url,
                    "doc_type": "html",
                    "page_title": page_title,
                    "element_type": "FallbackText",
                    "lang": "ita",
                    "crawl_ts": crawl_ts,
                    "doc_id": sha(source_url),
                },
            ))
        return docs

    for el in elements:
        text = getattr(el, "text", None) or str(el).strip()
        if not text:
            continue

        element_type = getattr(el, "category", None) or el.__class__.__name__
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}

        doc = Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "element_type": element_type,
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        )
        docs.append(doc)
    return docs


def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    """
    Se il link contiene 'orario', esegue SOLO l'estrazione dei JSON dell'orario.
    Altrimenti esegue la normale estrazione di testo + tabelle.
    """
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    # === Caso speciale: link contenente "orario" ===
    if "orario" in source_url.lower():
        print(f"[INFO] PDF orario rilevato → estrazione solo JSON: {source_url}")
        tables = extract_tables_camelot(file_path)

        if not tables:
            print(f"[WARN] Nessuna tabella trovata in {file_path.name}")
            return []

        # Raggruppa le tabelle per pagina
        page_groups = {}
        for t in tables:
            page_groups.setdefault(t["page_number"], []).append(t)

        for page_num, page_tables in page_groups.items():
            schedule_json = build_schedule_json(records=page_tables)
            if not schedule_json:
                continue

            # unisci slot consecutivi
            schedule_json = merge_consecutive_lessons(schedule_json)

            # aggiungi contesto
            year = "I" if page_num == 1 else "II"
            semester = "I"
            period = "15/09/2025 - 12/12/2025"

            enriched_json = {
                "corso_di_laurea": "Ingegneria Informatica e Robotica",
                "anno_accademico": "2025/2026",
                "anno": year,
                "semestre": semester,
                "periodo": period,
                "orario": schedule_json
            }

            docs.append(Document(
                page_content=json.dumps(enriched_json, ensure_ascii=False, indent=2),
                metadata={
                    "source_url": source_url,
                    "doc_type": "pdf-schedule",
                    "page_number": page_num,
                    "year": year,
                    "semester": semester,
                    "period": period,
                    "file_name": file_path.name,
                    "element_type": "ScheduleJSON",
                    "crawl_ts": crawl_ts,
                    "doc_id": sha(f"{source_url}_{page_num}")
                }
            ))
        return docs

    # === Caso normale: PDF qualsiasi ===
    elements = []
    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
            infer_table_structure=True,
            languages=["ita", "eng"]
        )
    except Exception as e:
        print(f"[WARN] partition_pdf fallito con strategy hi_res su {file_path}: {e}")

    for el in elements:
        text = getattr(el, "text", None) or str(el).strip()
        if not text:
            continue
        element_type = getattr(el, "category", None) or el.__class__.__name__
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}

        docs.append(Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "pdf",
                "page_number": meta.get("page_number", None),
                "file_name": file_path.name,
                "element_type": element_type,
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        ))

    # Estrazione tabelle normali (non orari)
    tables = extract_tables_camelot(file_path)
    if tables:
        for t in tables:
            docs.append(Document(
                page_content=json.dumps(t["row"], ensure_ascii=False),
                metadata={
                    "source_url": source_url,
                    "doc_type": "pdf-table",
                    "page_number": t["page_number"],
                    "file_name": t["file_name"],
                    "table_index": t["table_index"],
                    "element_type": "Table",
                    "crawl_ts": crawl_ts,
                    "doc_id": sha(source_url)
                }
            ))

    return docs


def extract_tables_camelot(file_path: Path) -> list[dict]:
    records = []
    try:
        tables = camelot.read_pdf(str(file_path), pages="all", flavor="lattice")
        if not tables or len(tables) == 0:
            tables = camelot.read_pdf(str(file_path), pages="all", flavor="stream")
        for t_idx, table in enumerate(tables):
            df = table.df
            for i in range(len(df)):
                row = df.iloc[i].to_dict()
                records.append({
                    "row": row,
                    "page_number": table.page,
                    "file_name": file_path.name,
                    "table_index": t_idx
                })
                # print(f"[DEBUG] riga tabella: {row}")
    except Exception as e:
        print(f"[WARN] Camelot fallito su {file_path}: {e}")
    return records

import re
import json

def build_schedule_json(records: list[dict]) -> dict:
    """
    Ricostruisce l'orario da una tabella PDF con colonne per i giorni e colonne "aule".
    Restituisce: { giorno -> { fascia_oraria -> [ { "corso": ..., "aula": ... } ] } }
    """
    import re

    if not records:
        return {}

    # Individua le colonne dei giorni e delle aule
    header_row = records[0]["row"]
    day_columns = {}
    room_columns = {}
    keys = list(header_row.keys())
    for i, v in header_row.items():
        day = v.strip().upper()
        if day in ["LUNEDÌ", "MARTEDÌ", "MERCOLEDÌ", "GIOVEDÌ", "VENERDÌ"]:
            day_columns[i] = day.capitalize()
            # subito dopo dovrebbe esserci la colonna "aule"
            if (i + 1) in header_row and "AULE" in header_row[i + 1].upper():
                room_columns[i + 1] = day.capitalize()

    schedule = {day.capitalize(): {} for day in day_columns.values()}

    # Scorri tutte le righe successive (quelle con orari + corsi + aule)
    for r in records[1:]:
        row = {k: v.strip().replace("\n", " ") for k, v in r["row"].items() if v.strip()}

        # Estrai la fascia oraria
        time_match = None
        for v in row.values():
            m = re.search(r"(\d{1,2}[:\.]\d{2})\s*[-–]?\s*(\d{1,2}[:\.]\d{2})", v)
            if m:
                start = m.group(1).replace(".", ":")
                end = m.group(2).replace(".", ":")
                time_match = f"{start}-{end}"
                break
        if not time_match:
            continue

        # Per ogni giorno, salva corso + aula
        for k, day in day_columns.items():
            course = row.get(k, "").strip().replace("\n", " ")
            room = row.get(k + 1, "").strip().replace("\n", " ") if (k + 1) in row else ""
            if course:
                schedule[day].setdefault(time_match, []).append({
                    "corso": course,
                    "aula": room
                })

    return schedule

from datetime import datetime

from datetime import datetime

def normalize_time(t: str) -> str:
    """Converte un orario tipo '14.30' o '14:30' in formato HH:MM con zeri."""
    t = t.replace(".", ":").strip()
    return datetime.strptime(t, "%H:%M").strftime("%H:%M")

def normalize_room(r: str) -> str:
    """Normalizza il nome dell'aula (spazi, maiuscole)."""
    return r.strip().upper()

import unicodedata
import re

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # normalizza unicode (es. spazi speciali → normali)
    s = unicodedata.normalize("NFKC", s)
    # togli ritorni a capo e spazi multipli
    s = re.sub(r"\s+", " ", s)
    return s.strip().upper()


def merge_consecutive_lessons(schedule: dict) -> dict:
    merged_schedule = {}

    for day, slots in schedule.items():
        # ordina per ora di inizio
        times = sorted(slots.keys(), key=lambda t: datetime.strptime(t.split("-")[0], "%H:%M"))
        merged_day = {}
        i = 0

        while i < len(times):
            start_time, end_time = times[i].split("-")
            lessons = slots[times[i]]

            if len(lessons) != 1:
                merged_day[f"{start_time}-{end_time}"] = lessons
                i += 1
                continue

            current_course = normalize_text(lessons[0]["corso"])
            current_room = normalize_room(lessons[0]["aula"])
            current_start = normalize_time(start_time)
            current_end = normalize_time(end_time)

            j = i + 1
            while j < len(times):
                next_start, next_end = times[j].split("-")
                next_lessons = slots[times[j]]

                if (
                    len(next_lessons) == 1
                    and normalize_text(next_lessons[0]["corso"]) == current_course
                    and normalize_room(next_lessons[0]["aula"]) == current_room
                    and normalize_time(next_start) == current_end  # continuità temporale
                ):
                    current_end = normalize_time(next_end)
                    j += 1
                else:
                    break

            merged_day[f"{current_start}-{current_end}"] = [
                {"corso": current_course, "aula": current_room}
            ]
            i = j

        merged_schedule[day] = merged_day

    return merged_schedule
