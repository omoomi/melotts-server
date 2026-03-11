from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LANGUAGE_ID_EN = 2
TONE_START_EN = 7

_ARPA = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G", "AY0",
    "M", "AA0", "F", "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0", "EY1", "IH0",
    "K", "N", "W", "IY2", "T", "AA1", "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2",
    "AW1", "V", "UW2", "AA2", "ER", "AW0", "UW0", "R", "OW1", "EH1", "ZH", "AE0",
    "IH2", "IH", "Y", "JH", "P", "AY1", "EY0", "OY2", "TH", "HH", "D", "ER0", "CH",
    "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0", "L", "SH",
}


def _intersperse(lst: list[int], item: int) -> list[int]:
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _distribute_phone(total_phones: int, wordpieces: int) -> list[int]:
    if wordpieces <= 0:
        return []
    base = total_phones // wordpieces
    rem = total_phones % wordpieces
    return [base + (1 if i < rem else 0) for i in range(wordpieces)]


def _normalize_en_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _post_replace_phone(ph: str, symbols: set[str]) -> str:
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "v": "V",
    }
    ph = rep_map.get(ph, ph)
    return ph if ph in symbols else "UNK"


@dataclass(slots=True)
class FrontendOutput:
    token_ids: np.ndarray
    tone_ids: np.ndarray
    lang_ids: np.ndarray
    bert: np.ndarray
    ja_bert: np.ndarray


class MeloEnglishFrontend:
    """Melo-style EN frontend for the converted English-v3 bundle."""

    def __init__(self, config_path: Path, frontend_assets: Path, use_bert: bool = True):
        self.use_bert = use_bert
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        self.add_blank = bool(cfg.get("data", {}).get("add_blank", True))

        symbols = cfg.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise RuntimeError("config.json missing valid symbols list")
        self.symbol_to_id = {str(s): i for i, s in enumerate(symbols)}
        self.symbol_set = set(self.symbol_to_id.keys())

        spk2id = cfg.get("data", {}).get("spk2id", {})
        self.valid_speaker_ids = sorted({int(v) for v in spk2id.values()}) if isinstance(spk2id, dict) else []

        # Torch occasionally sees a stripped logging name map in non-interactive startup shells.
        # Re-register the standard names before importing transformers/torch.
        logging.addLevelName(logging.CRITICAL, "CRITICAL")
        logging.addLevelName(logging.ERROR, "ERROR")
        logging.addLevelName(logging.WARNING, "WARNING")
        logging.addLevelName(logging.INFO, "INFO")
        logging.addLevelName(logging.DEBUG, "DEBUG")

        try:
            from g2p_en import G2p
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing English frontend deps. Install g2p_en and transformers in the melotts-server environment."
            ) from exc

        self._g2p = G2p()
        model_root = str(frontend_assets.resolve())
        self._tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        self._bert_model = None
        if self.use_bert:
            self._bert_model = AutoModelForMaskedLM.from_pretrained(model_root, local_files_only=True)
            self._bert_model.eval()

    def _tokenize_to_phones(self, norm_text: str) -> tuple[list[int], list[int], list[int]]:
        tokenized = self._tokenizer.tokenize(norm_text)
        groups: list[list[str]] = []
        for tok in tokenized:
            if not tok.startswith("#"):
                groups.append([tok])
            elif groups:
                groups[-1].append(tok.replace("#", ""))
            else:
                groups.append([tok.replace("#", "")])

        phones: list[str] = []
        tones: list[int] = []

        for group in groups:
            word = "".join(group)
            raw_phones = [p for p in self._g2p(word) if p != " "]
            group_phone_count = 0
            for ph in raw_phones:
                if ph in _ARPA:
                    tone = int(ph[-1]) + 1 if ph[-1].isdigit() else 0
                    ph = ph[:-1] if ph[-1].isdigit() else ph
                    phones.append(_post_replace_phone(ph.lower(), self.symbol_set))
                    tones.append(tone)
                else:
                    phones.append(_post_replace_phone(ph, self.symbol_set))
                    tones.append(0)
                group_phone_count += 1

            if group_phone_count == 0:
                phones.append("_")
                tones.append(0)
                group_phone_count = 1

            _distribute_phone(group_phone_count, len(group))

        phone_ids = [self.symbol_to_id.get(p, self.symbol_to_id.get("UNK", 0)) for p in phones]
        tone_ids = [t + TONE_START_EN for t in tones]
        lang_ids = [LANGUAGE_ID_EN] * len(phone_ids)
        return phone_ids, tone_ids, lang_ids

    def _build_bert(self, norm_text: str, token_count: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_bert:
            return np.zeros((1024, token_count), dtype=np.float32), np.zeros((768, token_count), dtype=np.float32)

        import torch

        with torch.no_grad():
            inputs = self._tokenizer(norm_text, return_tensors="pt")
            res = self._bert_model(**inputs, output_hidden_states=True)
            last = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu().numpy()

        repeats = max(token_count, 1)
        aligned = np.repeat(last[:1, :], repeats, axis=0).T.astype(np.float32)
        return np.zeros((1024, token_count), dtype=np.float32), aligned

    def encode(self, text: str) -> FrontendOutput:
        norm_text = _normalize_en_text(text)
        phone_ids, tone_ids, lang_ids = self._tokenize_to_phones(norm_text)
        if self.add_blank:
            phone_ids = _intersperse(phone_ids, 0)
            tone_ids = _intersperse(tone_ids, 0)
            lang_ids = _intersperse(lang_ids, 0)

        t = len(phone_ids)
        bert, ja_bert = self._build_bert(norm_text, t)
        return FrontendOutput(
            token_ids=np.asarray(phone_ids, dtype=np.int64),
            tone_ids=np.asarray(tone_ids, dtype=np.int64),
            lang_ids=np.asarray(lang_ids, dtype=np.int64),
            bert=bert,
            ja_bert=ja_bert,
        )
