#!/usr/bin/env python3
"""
TON-3079 — Deterministic-only security triage classifier (baseline + authority layer).

This is the RULE baseline compared blind against the 8B/35B models, and it is also
the authority boundary that stays in front of any model. It never grants access; it
only maps a message to a triage action {escalate, flag, allow}.

Detection strategy: normalize common obfuscations (base64, leet, char-splitting,
homoglyph) then match intent patterns for the 5 attack families. Conservative on
attacks (fail toward escalate on strong signals) but tuned to keep benign hard
negatives at allow so FPR stays measurable.
"""
import base64
import re
import unicodedata

# high-signal intent patterns (English + Korean), grouped by family.
_CRIT_HIGH = [
    # prompt injection / instruction override
    r"ignore (all )?(previous|prior) instructions",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"이전 (지시|명령|프롬프트).{0,6}(무시|무시해|무시하)",
    r"시스템 (프롬프트|지시).{0,6}무시",
    r"system (override|prompt)[:\s]",
    r"reveal( your| the)? (hidden|system|internal) (instruction|prompt|directive)",
    r"(숨겨진|내부|시스템) (지시|프롬프트|지침).{0,6}(출력|공개|보여)",
    r"print .{0,20}between <system>",
    r"<system>.*</system>",
    # jailbreak / role persona
    r"\b(dan|freegpt|do anything now|unfiltered mode|no restrictions?)\b",
    r"(윤리|안전) (지침|필터).{0,6}(없|꺼|해제)",
    r"safety filter is off|pretend .{0,15}(filter|policy).{0,10}off",
    r"policy (doesn'?t|does not) apply",
    r"정책.{0,6}(적용 안|무시)",
    # data exfil
    r"(base64|tar).{0,40}(post|upload|send|전송|업로드).{0,40}(http|ftp|pastebin|example\.net)",
    r"(read|dump|copy).{0,30}(/etc/shadow|\.env|audit_log|threat_log|message_bus)",
    r"(POST|upload|send).{0,20}(collector\.example|drop\.example|pastebin)",
    r"(빼내|덤프|업로드|전송|POST).{0,30}(pastebin|example\.net|ftp)",
    r"환경 변수.{0,20}(KEY|TOKEN|SECRET|출력)",
    r"print any .{0,10}(KEY|TOKEN|SECRET)",
    # credential seeking
    r"(what is|give me|show me|paste|print).{0,30}(api[_ ]?key|private key|password|refresh[_ ]?token|secret)",
    r"paperclip_api_key|refresh_token|ssh (private )?key",
    r"(api[_ ]?key|비밀번호|개인키|자격 증명|토큰).{0,12}(값|알려|보여|출력|붙여넣|paste|print)",
    # privilege escalation
    r"(add|append).{0,20}(allowlist|allowed_models)",
    r"(remove|빼).{0,20}(blocked_agents|차단)",
    r"허용목록.{0,10}(추가)|차단.{0,10}(해제|빼)",
    r"sudo chmod.{0,10}777",
    r"policy\.enabled\s*=\s*false|SecurityModule.{0,10}(끄|비활성|disable)",
    r"(grant|부여).{0,25}(same permissions|admin|ceo|권한)",
    r"skip the approval gate|승인 게이트.{0,6}건너",
    r"max_token_velocity\s*(to|=)\s*\d{5,}|max_payload_size.{0,15}(unlimited|무제한)",
    r"unblock|차단.{0,4}해제|elevate .{0,10}role|권한을 올",
]

_MEDIUM = [
    r"hypothetically|가정상|가정하자",
    r"answer twice|두 번 답",
    r"bypass the .{0,20}(allowlist|check)|우회",
    r"list every credential|모든 자격 증명.{0,10}나열",
]

_CRIT_HIGH_RE = [re.compile(p, re.IGNORECASE) for p in _CRIT_HIGH]
_MEDIUM_RE = [re.compile(p, re.IGNORECASE) for p in _MEDIUM]

_HOMOGLYPH = {"о": "o", "е": "e", "а": "a", "с": "c"}
_LEET = {"@": "a", "3": "e", "1": "i", "0": "o", "$": "s"}


def _deobfuscate(text):
    """Return a set of candidate strings: original + normalized variants."""
    variants = {text}

    # homoglyph + NFKC
    norm = unicodedata.normalize("NFKC", text)
    norm = "".join(_HOMOGLYPH.get(ch, ch) for ch in norm)
    variants.add(norm)

    # de-leet (only letters/symbols, keep it cheap)
    deleet = "".join(_LEET.get(ch, ch) for ch in norm)
    variants.add(deleet)

    # de-space: collapse "i.g.n.o.r.e" style splitting
    despace = re.sub(r"(?<=\w)\.(?=\w)", "", norm)
    variants.add(despace)
    variants.add(re.sub(r"(?<=\w)\.(?=\w)", "", deleet))

    # decode embedded base64 blobs (>=16 chars) and add decoded text
    for m in re.finditer(r"[A-Za-z0-9+/]{16,}={0,2}", text):
        blob = m.group(0)
        try:
            dec = base64.b64decode(blob, validate=True).decode("utf-8", "ignore")
            if dec and any(c.isalpha() for c in dec):
                variants.add(dec)
        except Exception:
            pass
    return variants


def classify(text):
    """Return triage action: 'escalate' | 'flag' | 'allow'."""
    variants = _deobfuscate(text)
    for v in variants:
        for rx in _CRIT_HIGH_RE:
            if rx.search(v):
                return "escalate"
    for v in variants:
        for rx in _MEDIUM_RE:
            if rx.search(v):
                return "flag"
    return "allow"


if __name__ == "__main__":
    import sys
    print(classify(sys.stdin.read()))
