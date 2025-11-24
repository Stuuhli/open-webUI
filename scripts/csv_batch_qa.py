#!/usr/bin/env python3
"""
Batch-Runner: liest Fragen aus einer CSV, fragt Open WebUI mit einer
Knowledge-Collection ab und schreibt die Antworten wieder in die CSV.

Voraussetzungen:
- Eine API-Key des angemeldeten Users (Profile -> API Keys)
- Die Knowledge-Collection-ID oder ihr Name
- requests ist installiert (im Backend bereits vorhanden)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import requests


def parse_multi(values: Sequence[str] | None) -> List[str]:
    """Zerlegt Mehrfach-Inputs (mehrfacher Flag oder Komma-getrennt)."""
    result: list[str] = []
    if not values:
        return result
    for value in values:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        result.extend(parts)
    # Reihenfolge beibehalten, Duplikate entfernen
    seen = set()
    unique: list[str] = []
    for item in result:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def fetch_knowledge_list(base_url: str, headers: Dict[str, str]) -> list[dict]:
    """
    Holt die Knowledge-Liste und versucht dabei sowohl Pfad mit als auch ohne Slash.
    Liefert bei HTML-Antwort eine klare Fehlermeldung, damit Frontend/Backend-Verwechslungen auffallen.
    """
    urls = [
        f"{base_url.rstrip('/')}/api/v1/knowledge",
        f"{base_url.rstrip('/')}/api/v1/knowledge/",
    ]

    last_resp = None
    for url in urls:
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        last_resp = resp
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception:
            # Weiter zum nächsten Versuch
            continue

    # Wenn kein Versuch erfolgreich war, eine gut lesbare Fehlermeldung ausgeben
    if last_resp is not None:
        body_preview = last_resp.text[:500] if hasattr(last_resp, "text") else ""
        raise SystemExit(
            f"Fehler beim Laden der Knowledge-Liste ({last_resp.status_code}): "
            f"Content-Type={last_resp.headers.get('content-type')}\n"
            f"Antwort-Ausschnitt: {body_preview}"
        )
    else:
        raise SystemExit("Fehler beim Laden der Knowledge-Liste: keine Antwort erhalten.")


def resolve_knowledge_ids(
    base_url: str,
    headers: Dict[str, str],
    names: List[str],
    knowledge_ids: List[str],
) -> List[str]:
    """Findet Knowledge-IDs per Name oder nutzt die übergebenen Werte."""
    resolved = list(knowledge_ids)

    if names:
        knowledge_list = fetch_knowledge_list(base_url, headers)
        available = {kb.get("name", "").lower(): kb["id"] for kb in knowledge_list}

        for name in names:
            kb_id = available.get(name.lower())
            if not kb_id:
                raise SystemExit(f"Knowledge '{name}' nicht gefunden.")
            resolved.append(kb_id)

    if not resolved:
        raise SystemExit("Bitte mindestens eine --knowledge-id oder --knowledge-name angeben.")

    # Duplikate entfernen, Reihenfolge beibehalten
    seen = set()
    unique: list[str] = []
    for item in resolved:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def ask_model(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    knowledge_ids: List[str],
    question: str,
    temperature: float | None,
    full_context: bool,
    timeout: int,
) -> str:
    files_payload = [
        {
            "type": "collection",
            "id": knowledge_id,
            **({"context": "full"} if full_context else {}),
        }
        for knowledge_id in knowledge_ids
    ]

    payload: Dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "files": files_payload,
        "stream": False,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    resp = requests.post(
        f"{base_url}/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Unerwartete Antwort: {data}")

    message = choices[0].get("message", {})
    return message.get("content", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fragen aus CSV per Open WebUI beantworten und als neue CSV speichern."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Pfad zur Eingabe-CSV")
    parser.add_argument(
        "--output",
        type=Path,
        help="Pfad zur Ausgabe-CSV (Standard: input_with_answers.csv)",
    )
    parser.add_argument(
        "--question-column",
        default="question",
        help="Spaltenname mit der Frage (Standard: question)",
    )
    parser.add_argument(
        "--answer-column",
        default="answer",
        help="Spaltenname, in die die Antwort geschrieben wird (Standard: answer)",
    )
    parser.add_argument("--model", required=True, help="Model-ID aus Open WebUI")
    parser.add_argument(
        "--knowledge-id",
        action="append",
        help="Knowledge/Collection-ID (mehrfach nutzbar oder Komma-getrennt). Alternative: --knowledge-name",
    )
    parser.add_argument(
        "--knowledge-name",
        action="append",
        help="Knowledge/Collection-Name (mehrfach nutzbar oder Komma-getrennt; wird aufgelöst, wenn keine ID angegeben).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENWEBUI_URL", "http://localhost:3000"),
        help="Basis-URL der Open WebUI API (Standard: http://localhost:3000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENWEBUI_API_KEY"),
        help="API-Key (sk-...). Kann auch über die Umgebungsvariable OPENWEBUI_API_KEY gesetzt werden.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optionaler Temperaturwert für das Modell.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Pause in Sekunden zwischen Anfragen (z. B. 0.5 bei Rate-Limits).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout pro Anfrage in Sekunden (Standard: 120).",
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Knowledge-Collection komplett als Kontext schicken (anstatt Vektor-Suche).",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Kein API-Key gefunden. Lege in deinem Profil einen API-Key an und setze ihn via --api-key oder OPENWEBUI_API_KEY."
        )

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    knowledge_ids = resolve_knowledge_ids(
        args.base_url,
        headers,
        parse_multi(args.knowledge_name),
        parse_multi(args.knowledge_id),
    )

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_with_answers{args.input.suffix}"
    )

    with args.input.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if args.question_column not in (reader.fieldnames or []):
            raise SystemExit(
                f"Spalte '{args.question_column}' nicht gefunden. Verfügbare Spalten: {reader.fieldnames}"
            )
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if args.answer_column not in fieldnames:
        fieldnames.append(args.answer_column)

    with output_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            question = (row.get(args.question_column) or "").strip()
            if not question:
                row[args.answer_column] = ""
                writer.writerow(row)
                continue

            try:
                answer = ask_model(
                    base_url=args.base_url.rstrip("/"),
                    headers=headers,
                    model=args.model,
                    knowledge_ids=knowledge_ids,
                    question=question,
                    temperature=args.temperature,
                    full_context=args.full_context,
                    timeout=args.timeout,
                )
                row[args.answer_column] = answer
                print(f"[{idx}/{len(rows)}] OK")
            except Exception as exc:  # noqa: BLE001
                row[args.answer_column] = f"FEHLER: {exc}"
                print(f"[{idx}/{len(rows)}] Fehler: {exc}", file=sys.stderr)

            writer.writerow(row)
            if args.pause > 0:
                time.sleep(args.pause)

    print(f"Fertig. Datei geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()
