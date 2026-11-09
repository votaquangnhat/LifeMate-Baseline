"""Vietnamese label normalization and input-only contextual phrase extraction."""
import re
import unicodedata
from pathlib import Path
import json

VERSION = 'spoken_vi_v4.1'
DIGITS = ('không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín')
DEFAULT_WORD_MAP = {'giùm':'dùm', 'huỷ':'hủy', 'xoá':'xóa', 'hoà':'hòa',
                    'hoá':'hóa', 'khoá':'khóa', 'khoẻ':'khỏe', 'thuỷ':'thủy'}

def norm(text):
    text = unicodedata.normalize('NFC', str(text)).lower()
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LMN' else ' ' for c in text).split())

def integer_words(value):
    n = int(value)
    if n < 0:
        return 'âm ' + integer_words(-n)
    if n < 10:
        return DIGITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        words = ['mười'] if tens == 1 else [DIGITS[tens], 'mươi']
        if unit:
            words.append('lăm' if unit == 5 else 'mốt' if unit == 1 and tens > 1 else DIGITS[unit])
        return ' '.join(words)
    if n < 1000:
        h, rest = divmod(n, 100)
        return DIGITS[h] + ' trăm' + ((' linh ' + DIGITS[rest]) if 0 < rest < 10 else ' ' + integer_words(rest) if rest else '')
    for scale, name in ((10**9,'tỷ'),(10**6,'triệu'),(1000,'nghìn')):
        if n >= scale:
            high, low = divmod(n, scale)
            return integer_words(high) + ' ' + name + (' ' + integer_words(low) if low else '')
    raise ValueError(value)

def digit_words(value):
    return ' '.join(DIGITS[int(c)] for c in value if c.isdigit())

def spoken_text(text, cfg=None, return_notes=False):
    """Fixed rules for labels; raw references stay separate. Ambiguities are logged.

    No global replacements of ni/mo/rang/ma, names, negatives or number words.
    Written times/dates have multiple valid readings: review normalization_audit.csv.
    """
    cfg = cfg or {}
    text = unicodedata.normalize('NFC', str(text)).lower()
    text = ''.join(str(unicodedata.digit(c)) if c.isdigit() and unicodedata.category(c)=='Nd' else c for c in text)
    notes = []
    protected = []
    def hold(value):
        protected.append(value)
        # Private-use characters cannot be confused with transcript digits.
        return chr(0xE000 + len(protected)-1)
    def phone(m):
        notes.append('phone_digits')
        return hold(digit_words(re.sub(r'\D','',m.group())))
    text = re.sub(r'(?<![\w\d])0(?:[ .\-]?\d){8,12}(?!\d)', phone, text)
    def clock(m):
        h, minute = int(m[1]), int(m[2])
        if not (0 <= h <= 23 and 0 <= minute <= 59):
            notes.append('review_invalid_clock'); return m.group()
        notes.append('review_clock_reading')
        return hold(integer_words(h) + ' giờ' + (' ' + integer_words(minute) + ' phút' if minute else ''))
    text = re.sub(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)', clock, text)
    def date(m):
        day, month, year = int(m[1]), int(m[2]), m[3]
        if not (1 <= day <= 31 and 1 <= month <= 12):
            notes.append('review_slash_number'); return m.group()
        prefix = text[max(0,m.start()-30):m.start()]
        if year or re.search(r'(?:ngày|mùng|hôm|lịch|hẹn|tháng)\b', prefix):
            notes.append('review_date_reading')
            first = '' if re.search(r'(?:ngày|mùng)\s*$', prefix) else 'ngày '
            return hold(first + integer_words(day) + ' tháng ' + integer_words(month) + (' năm ' + integer_words(year) if year else ''))
        notes.append('review_fraction_reading')
        return hold(integer_words(day)+' phần '+integer_words(month))
    text = re.sub(r'(?<!\d)(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?!\d)', date, text)
    # Vietnamese thousands grouping; comma decimals remain decimals.
    text = re.sub(r'(?<!\d)\d{1,3}(?:\.\d{3})+(?!\d)', lambda m: hold(integer_words(m.group().replace('.',''))), text)
    def decimal(m):
        notes.append('review_decimal_reading')
        return hold(integer_words(m[1])+' phẩy '+digit_words(m[2]))
    text = re.sub(r'(?<!\d)(\d+)[.,](\d+)(?!\d)', decimal, text)
    text = re.sub(r'(?<!\w)-(\d+)',lambda m:hold('âm '+integer_words(m[1])),text)
    text = re.sub(r'(\d+)\s*[–—]\s*(\d+)',lambda m:hold(integer_words(m[1])+' đến '+integer_words(m[2])),text)
    def number(m):
        value=m.group(); before=text[max(0,m.start()-30):m.start()]
        code = bool(before and re.search(r'[a-z]$',before))
        code = code or bool(re.search(r'(?:phòng|mã|chuyến|cơm tấm)\s*$',before))
        if (len(value)>1 and value.startswith('0')) or (code and len(value)>=3):
            notes.append('review_code_digits'); result=digit_words(value)
        else:
            result=integer_words(value)
        return ' '+result+' '
    text = re.sub(r'\d+', number, text)
    text = text.replace('%',' phần trăm ').replace('₫',' đồng ')
    for i, value in enumerate(protected):text=text.replace(chr(0xE000+i),' '+value+' ')
    text=norm(text)
    mapping=cfg.get('word_map',DEFAULT_WORD_MAP)
    text=' '.join(mapping.get(word,word) for word in text.split())
    if re.search(r'\d',text):raise ValueError('Unconverted digits after spoken normalization: '+text)
    return (text,sorted(set(notes))) if return_notes else text

def focus_tags(text):
    low=norm(text)
    tags=[]
    if re.search(r'\d',text) or re.search(r'\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|lăm|mươi)\s+(?:giờ|phút|độ|triệu|nghìn|ngàn|trăm)\b',low):tags.append('numbers')
    if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b',text) or re.search(r'\b[A-Z]{2,}\d*\b',text):tags.append('name_or_code')
    if re.search(r'\b(?:nếu|trừ|đừng)\b',low) or re.search(r'\bkhông\b.*\bthì\b',low):tags.append('condition')
    return tags

ENTITY_KEYS={'name','full_name','display_name','contact_name','room_name','title','artist','author',
             'venue','brand','alias','aliases','recipient_name','location_name','address'}
FORBIDDEN={'label','labels','gold','reference','answer','final_state','tool_calls','tool_results','audio_text','transcript','history_audio_text','history_transcripts'}
def context_phrases(item, base=None, env_prefix='train', limit=128):
    """Only supplied context and prior history; never hidden environment state or gold."""
    phrases=set()
    def add(value):
        value=' '.join(value.split()).strip()
        if 2<=len(value)<=80 and 1<=len(norm(value).split())<=8 and any(c.isalpha() for c in value):phrases.add(value)
    def walk(value,key='',depth=0):
        if depth>12:return
        if isinstance(value,dict):
            for k,v in value.items():
                if k.lower() not in FORBIDDEN:walk(v,k.lower(),depth+1)
        elif isinstance(value,list):
            for v in value:walk(v,key,depth+1)
        elif isinstance(value,str) and key in ENTITY_KEYS:add(value)
    walk(item.get('context',{}))
    for turn in item.get('history',[]):
        if turn.get('role') == 'assistant':
            walk(turn.get('tool_results', []))
        # Already-provided text messages only; do not consume ASR reference fields.
        value=turn.get('content') or turn.get('text')
        if isinstance(value,str):
            for match in re.finditer(r'\b[A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){1,5}\b',value):add(match.group())
    return sorted(phrases,key=lambda s:(-len(s.split()),s.casefold()))[:limit]
