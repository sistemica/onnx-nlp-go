"""Renders chat templates with jinja2 the way transformers does and records the results.

usage: python golden.py OUT.json NAME=tokenizer_config.json ... [--snippets]

The Go tests compare ChatTemplate.Render with these outputs.
"""
import json
import sys

import jinja2
import jinja2.ext
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

FIXED_DATE = "26 Jul 2024"


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise TemplateError(message)


def environment():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: FIXED_DATE
    return env


def msg(role, content):
    return {"role": role, "content": content}


CONVERSATIONS = {
    "single question": [msg("user", "Wie viele Beine hat eine Spinne?")],
    "with system": [msg("system", "You are terse."), msg("user", "Hi")],
    "two turns": [
        msg("system", "Be brief."),
        msg("user", "What is 2+2?"),
        msg("assistant", "4"),
        msg("user", "And 3+3?"),
    ],
    "no system, three turns": [
        msg("user", "a"),
        msg("assistant", "b"),
        msg("user", "c"),
        msg("assistant", "d"),
        msg("user", "e"),
    ],
    "padding and unicode": [msg("system", "  spaced \n"), msg("user", "  héllo 😀\n\n"), msg("assistant", " ok  ")],
    "assistant first": [msg("assistant", "hello"), msg("user", "hi")],
    "two users in a row": [msg("user", "a"), msg("user", "b")],
}


def token(config, key):
    value = config.get(key)
    if isinstance(value, dict):
        value = value.get("content")
    return value


def template_of(config):
    template = config["chat_template"]
    if isinstance(template, list):
        template = next(t["template"] for t in template if t["name"] == "default")
    return template


def render(env, source, variables):
    try:
        return {"output": env.from_string(source).render(**variables)}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def model_cases(env, name, path):
    config = json.load(open(path))
    source = template_of(config)
    tokens = {k: token(config, k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
    tokens = {k: v for k, v in tokens.items() if v}
    cases = []
    for title, messages in CONVERSATIONS.items():
        for generation in (True, False):
            variables = dict(tokens, messages=messages, add_generation_prompt=generation)
            cases.append(dict(title=title, generation=generation, messages=messages, **render(env, source, variables)))
    return {"name": name, "template": source, "tokens": tokens, "cases": cases}


CONTEXT = {
    "messages": [msg("system", "S"), msg("user", "U"), msg("assistant", "A")],
    "xs": [3, 1, 2],
    "words": ["b", "a", "c"],
    "d": {"a": 1, "b": [1, 2], "c": {"x": None}},
    "s": "  Hello World  ",
    "n": 7,
    "neg": -7,
    "f": 2.5,
    "nothing": None,
    "yes": True,
    "no": False,
    "empty": "",
    "text": "one,two,,three",
    "people": [{"name": "Ann", "age": 31, "tags": ["a"]}, {"name": "Bob", "age": 25, "tags": []}, {"name": "Cy", "age": 31}],
}

SNIPPETS = [
    "{{ 1 }}", "{{ 'a' ~ 1 ~ none }}", "{{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 7 / 2 }} {{ 2 ** 10 }} {{ 2 ** -1 }}",
    "{{ 7.0 // 2 }} {{ -7.5 % 2 }} {{ 1e3 }} {{ 1.5e-7 }} {{ 123456789012345678.0 }} {{ 0.1 + 0.2 }}",
    "{{ n + 1 }} {{ n - 10 }} {{ n * 2 }} {{ f * 2 }} {{ -n }} {{ +n }}", "{{ 'ab' * 3 }}{{ 2 * 'ab' }}", "{{ [1] * 3 }} {{ [1] + [2, 3] }}",
    "{{ 'a' + 'b' }}", "{{ 'a' + 1 }}", "{{ 1 / 0 }}", "{{ 5 % 0 }}",
    "{{ none }} {{ true }} {{ false }} {{ True }} {{ None }}", "{{ [1, 'a', none, true, 2.0, [3]] }}", "{{ {'a': 1, 'b': [1, 2]} }}",
    "{{ \"it's\" }} {{ ['it\\'s', 'say \"x\"', 'both \\' \"', 'tab\\t', 'nl\\n'] }}",
    "{{ 'a\\nb' }}|{{ 'a\\tb' }}|{{ 'back\\\\slash' }}|{{ '\\u00e9\\x41' }}|{{ '\\q' }}",
    "{{ 'adjacent' 'strings' }}",
    "{{ x }}|{{ x.y }}", "{{ d.missing }}|{{ d.missing.deeper }}", "{{ d['a'] }} {{ d.a }} {{ d['b'][1] }} {{ d.b[-1] }} {{ d.c.x }}",
    "{{ xs[0] }} {{ xs[-1] }} {{ xs[5] }}|{{ xs[1:] }} {{ xs[:-1] }} {{ xs[::-1] }} {{ xs[::2] }} {{ xs[1:2] }} {{ xs[-2:] }} {{ xs[5:] }}",
    "{{ s[2] }} {{ s[2:7] }} {{ s[::-1] }} {{ 'héllo'[1] }}", "{{ nothing.attr }}", "{{ nothing[0] }}", "{{ n.attr }}",
    "{{ n is defined }} {{ missing is defined }} {{ missing is not defined }} {{ missing is undefined }} {{ nothing is none }} {{ n is not none }}",
    "{{ s is string }} {{ n is number }} {{ n is integer }} {{ f is float }} {{ f is integer }} {{ yes is boolean }} {{ d is mapping }} {{ xs is iterable }} {{ n is iterable }} {{ xs is sequence }} {{ s is sequence }}",
    "{{ n is odd }} {{ n is even }} {{ 8 is divisibleby 4 }} {{ 8 is divisibleby(3) }} {{ n is eq 7 }} {{ n is ne 7 }} {{ n is lt 8 }} {{ n is gt 8 }} {{ n is le 7 }} {{ n is ge 8 }} {{ yes is true }} {{ no is false }} {{ n is in([7, 8]) }}",
    "{{ not n is defined }} {{ not missing is defined }}", "{{ n is nosuchtest }}", "{{ n|nosuchfilter }}",
    "{{ 1 < 2 }} {{ 2 <= 2 }} {{ 'a' < 'b' }} {{ 1 == 1.0 }} {{ 'a' == 'a' }} {{ 1 != 2 }} {{ [1, 2] == [1, 2] }} {{ {'a': 1} == {'a': 1} }} {{ none == none }} {{ true == 1 }}",
    "{{ 1 < 2 < 3 }} {{ 1 < 3 < 2 }} {{ 3 > 2 > 1 }} {{ 1 == 1 != 2 }}", "{{ 1 < 'a' }}", "{{ none < 1 }}",
    "{{ 'a' in 'abc' }} {{ 'z' in 'abc' }} {{ 2 in xs }} {{ 'a' in d }} {{ 'z' not in d }} {{ 5 not in xs }} {{ 'a' in missing }}", "{{ 1 in 'abc' }}", "{{ 1 in n }}",
    "{{ yes and n }} {{ no and n }} {{ no or n }} {{ n or no }} {{ nothing or 'x' }} {{ empty or 'fallback' }} {{ not empty }} {{ not xs }}",
    "{{ 'y' if yes else 'n' }} {{ 'y' if no else 'n' }}|{{ ('y' if no) }}|{{ 'a' if no else 'b' if yes else 'c' }}",
    "{{ (1, 2)|list }} {{ ()|list }} {{ (1,)|list }} {{ [] }} {{ {} }}", "{{ [1, 2, ] }} {{ {'a': 1, } }}",
    "{{ s|trim }}|{{ s|upper }}|{{ s|lower }}|{{ s|title }}|{{ 'hELLO wORLD'|capitalize }}|{{ s|length }}|{{ s|trim|length }}",
    "{{ xs|length }} {{ d|length }} {{ xs|count }} {{ missing|length }}", "{{ n|length }}", "{{ xs|first }} {{ xs|last }} {{ xs|reverse|list }} {{ 'abc'|reverse }} {{ empty|first }}|",
    "{{ xs|sort }} {{ words|sort }} {{ xs|sort(reverse=true) }} {{ people|sort(attribute='name')|map(attribute='name')|list }}", "{{ [1, 'a']|sort }}",
    "{{ xs|sum }} {{ [1.5, 2]|sum }} {{ words|join }} {{ words|join(', ') }} {{ xs|join('-') }} {{ people|join(',', attribute='name') }}",
    "{{ 'abc'|list }} {{ d|list }} {{ xs|list }} {{ d|items|map('first')|list }}", "{{ missing|default('dflt') }} {{ n|default('dflt') }} {{ empty|default('dflt') }} {{ empty|default('dflt', true) }} {{ missing|d('x') }} {{ missing|default }}|",
    "{{ '5'|int }} {{ 'x'|int }} {{ 'x'|int(3) }} {{ 2.9|int }} {{ '2.9'|int }} {{ '2.5'|float }} {{ 'x'|float }} {{ n|float }} {{ -3|abs }} {{ -2.5|abs }}",
    "{{ n|string }}|{{ xs|string }}|{{ nothing|string }}|{{ d|string }}|{{ yes|string }}", "{{ 'abc'|replace('b', 'X') }} {{ 'aaa'|replace('a', 'b', 2) }} {{ n|replace('7', 'x') }}",
    "{{ s|safe }}", "{{ 'a\\nb\\n\\nc'|indent(2) }}|{{ 'a\\nb'|indent(2, true) }}|{{ 'a\\n\\nb'|indent(width=2, blank=true) }}",
    "{{ d|tojson }} {{ xs|tojson }} {{ s|tojson }} {{ nothing|tojson }} {{ yes|tojson }} {{ f|tojson }} {{ 'é\\u0001\\\"'|tojson }} {{ {}|tojson }} {{ []|tojson }}",
    "{{ d|tojson(indent=2) }}|{{ xs|tojson(indent=4) }}|{{ {'a': {}, 'b': []}|tojson(indent=1) }}|{{ people|tojson }}",
    "{{ people|selectattr('age')|map(attribute='name')|list }} {{ people|rejectattr('age')|map(attribute='name')|list }} {{ people|selectattr('age', 'equalto', 31)|map(attribute='name')|list }} {{ people|selectattr('tags')|list|length }} {{ people|selectattr('name', 'in', ['Ann', 'Bob'])|list|length }} {{ people|rejectattr('age', 'gt', 30)|list|length }}",
    "{{ people|map(attribute='tags', default='none')|list }} {{ words|map('upper')|list }} {{ words|map('replace', 'a', 'z')|join }}",
    "{{ s.strip() }}|{{ s.lstrip() }}|{{ s.rstrip() }}|{{ 'xxhixx'.strip('x') }}|{{ s.upper() }}|{{ s.lower() }}|{{ 'a b'.title() }}|{{ 'aB'.capitalize() }}",
    "{{ text.split(',') }} {{ text.split(',', 1) }} {{ '  a  b c '.split() }} {{ 'a b c'.split(none, 1) }} {{ text.split() }}", "{{ 'abc'.split('') }}",
    "{{ 'abc'.startswith('ab') }} {{ 'abc'.endswith('bc') }} {{ 'abc'.startswith(('x', 'a')) }}", "{{ 'abc'.startswith(1) }}",
    "{{ 'abc'.replace('b', 'X') }} {{ ','.join(['a', 'b']) }}", "{{ 'abc'.nosuch() }}",
    "{{ d.items()|map('last')|list }} {{ d.keys()|list }} {{ d.values()|list }} {{ d.get('a') }} {{ d.get('z') }} {{ d.get('z', 'dflt') }} {{ d.get('z') is none }}",
    "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}", "{% for k in d %}{{ k }},{% endfor %}", "{% for c in 'abc' %}{{ c }}-{% endfor %}",
    "{% for i in range(3) %}{{ i }}{% endfor %}{% for i in range(1, 4) %}{{ i }}{% endfor %}{% for i in range(10, 0, -4) %}{{ i }},{% endfor %}",
    "{% for x in xs %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.length }}/{{ loop.first }}/{{ loop.last }};{% endfor %}",
    "{% for x in xs %}{{ loop.previtem }}|{{ loop.nextitem }};{% endfor %}", "{% for x in xs %}{{ x }}{% else %}empty{% endfor %}|{% for x in [] %}{{ x }}{% else %}empty{% endfor %}",
    "{% for x in xs if x > 1 %}{{ x }}:{{ loop.index }}:{{ loop.length }};{% endfor %}", "{% for x in missing %}{{ x }}{% else %}none{% endfor %}", "{% for x in n %}{{ x }}{% endfor %}", "{% for x in nothing %}{{ x }}{% endfor %}",
    "{% for a, b in [[1, 2], [3, 4]] %}{{ a + b }},{% endfor %}", "{% for a, b in [[1, 2, 3]] %}{{ a }}{% endfor %}", "{% for (a, b) in [[1, 2]] %}{{ a }}{{ b }}{% endfor %}",
    "{% for x in xs %}{% if x == 1 %}{% continue %}{% endif %}{{ x }}{% endfor %}", "{% for x in xs %}{% if x == 1 %}{% break %}{% endif %}{{ x }}{% endfor %}",
    "{% for x in xs %}{% for y in words %}{{ x }}{{ y }}{% if loop.last %}!{% endif %} {% endfor %}{% endfor %}",
    "{% for x in xs %}{% set last = x %}{% endfor %}[{{ last }}]", "{% set a = 1 %}{% for x in xs %}{% set a = x %}{{ a }}{% endfor %}{{ a }}",
    "{% set ns = namespace(total=0, seen=[]) %}{% for x in xs %}{% set ns.total = ns.total + x %}{% endfor %}{{ ns.total }} {{ ns.seen }} {{ ns.nope }}",
    "{% set ns = namespace() %}{% set ns.a = 1 %}{{ ns.a }}{% set ns.a = ns.a + 1 %}{{ ns.a }}", "{% set a = 1 %}{% set a.b = 2 %}",
    "{% set x = 5 %}{% if x > 3 %}{% set y = 'big' %}{% endif %}{{ y }}", "{% set a = [1, 2] %}{% set b = a + [3] %}{{ b }}{{ a }}", "{% set x = d|tojson %}{{ x }}", "{% set messages = messages[1:] %}{{ messages|length }}",
    "{% if n > 5 %}big{% elif n > 2 %}mid{% else %}small{% endif %}", "{% if n < 5 %}big{% elif n > 6 %}mid{% else %}small{% endif %}", "{% if n < 5 %}big{% elif n < 6 %}mid{% else %}small{% endif %}",
    "{% if missing %}a{% endif %}[{% if empty %}b{% else %}c{% endif %}]", "{% if missing.attr %}a{% endif %}", "{% if d.c.x is none and d.a == 1 %}ok{% endif %}", "{% if (n > 3) == (n < 9) %}ok{% endif %}",
    "{{ dict(a=1, b=2) }} {{ dict(d) }} {{ namespace(x=1).x }}", "{{ raise_exception('boom') }}", "{{ range(3)|list }} {{ range(2, 5)|list }}", "{{ range(0, 5, 0) }}",
    "{{ strftime_now('%d %b %Y') }}|{{ strftime_now('%Y-%m-%d %H:%M:%S %A %B %j %%') }}",
    "a  {# comment #}  b", "a {#- comment -#} b", "a\n{# comment #}\nb", "a {{- 1 -}} b", "a {{ 1 -}}\n  b", "a\n  {%- if yes -%}  x  {%- endif -%}\nb",
    "a\n  {% if yes %}\n  x\n  {% endif %}\nb", "a\n{% if yes %}\nx\n{% endif %}\nb\n", "a {% if yes %}x{% endif %} b", "  {% if yes %}x{% endif %}", "a\n    {{ 1 }}\n  {% if yes %}x{% endif %}",
    "line1\n{% for x in xs %}\n  {{ x }}\n{% endfor %}\nlast\n", "{%+ if yes %}x{% endif %}", "a\n   {%+ if yes %}x{% endif %}", "x\n\n\n{% set a = 1 %}\n\n\ny", "text\n", "text\n\n", "{% if yes %}\n{% endif %}\n",
    "{{ '}}' }}{{ '{%' }}{{ \"{{\" }}{{ {'a': {'b': 1}} }}", "{% set x = {'a': {'b': 1}} %}{{ x }}", "{{ '%}' }}", "{# a ' quote #}{{ 1 }}", "{{ 'a' }}{#- c -#}  {{ 'b' }}",
    "{{ n if n }}|{{ n if missing }}|", "{{ ('a', 'b')|join }}", "{{ d.b|length }} {{ d['c']['x'] is none }}", "{{ messages[0]['role'] }} {{ messages[-1].content }} {{ messages|length }} {{ messages[1:]|length }}",
    "{{ 'a' if messages[0]['role'] == 'system' else 'b' }}", "{{ messages|selectattr('role', 'equalto', 'user')|list|length }}", "{% for m in messages %}{{ m.role }}{% if not loop.last %},{% endif %}{% endfor %}",
    "{{ (messages|selectattr('role', 'equalto', 'user')|list)[-1].content }}", "{% if messages[0].role == 'system' %}{% set rest = messages[1:] %}{% else %}{% set rest = messages %}{% endif %}{{ rest|length }}",
    "{% if %}", "{% if n %}", "{% for %}x{% endfor %}", "{% for x in xs %}", "{% endif %}", "{{ ", "{{ 1 +* 2 }}", "{% macro f() %}{% endmacro %}[{{ f() }}]",
    "{% macro greet(name, punct='!') %}Hi {{ name }}{{ punct }}{% endmacro %}{{ greet('a') }}|{{ greet('b', '?') }}|{{ greet(name='c') }}|{{ greet('d', punct='.') }}",
    "{% macro f(x) %}\n  [{{ x }}]\n{% endmacro %}{{ f(1) }}{{ f(2) }}", "{% macro f(x) -%}\n  [{{ x }}]\n{%- endmacro %}{{ f(1) }}{{ f(2) }}",
    "{% macro fact(n) %}{% if n <= 1 %}1{% else %}{{ n * fact(n - 1)|int }}{% endif %}{% endmacro %}{{ fact(5) }}",
    "{% set greeting = 'hi' %}{% macro f() %}{{ greeting }}{% set greeting = 'inner' %}{{ greeting }}{% endmacro %}{{ f() }}{{ greeting }}",
    "{% macro f(a, b) %}{{ a }}{{ b }}{% endmacro %}{{ f(1) }}", "{% macro f(a) %}{{ a }}{% endmacro %}{{ f(1, 2) }}", "{% macro f(a) %}{{ a }}{% endmacro %}{{ f(b=2) }}", "{% macro f(a) %}{{ a }}",
    "{% macro f(a) %}{{ a|upper }}{% endmacro %}{{ f('x') ~ f('y') }}{{ f(d.b)|length }}", "{% macro f() %}x{% endmacro %}{% if f() == 'x' %}same{% endif %}",
    "{% set x %}a{{ n }}b{% endset %}[{{ x }}]", "{% set x %}\n  v\n{% endset %}[{{ x }}]", "{% set x %}a", "{{ -n|abs }} {{ -n|string }} {{ -xs[0] }} {{ -d.a }} {{ -f|abs }}", "{% set %}", "{{ [1, 2 }}", "{% if n %}x{% else %}y{% else %}z{% endif %}", "{{ 'unterminated }}", "{% include 'x' %}",
    "{{ -n|abs }} {{ (-n)|abs }} {{ n|abs|string + 'x' }}", "{{ 1 + 2 * 3 - 4 / 2 }} {{ (1 + 2) * 3 }} {{ 2 * 3 ** 2 }} {{ 10 - 2 - 3 }} {{ 2 ** 3 ** 2 }}", "{{ 'a' ~ 1 + 2 }}", "{{ 1 ~ 2 + 3 }}",
    "{{ not 1 == 2 }} {{ not (1 == 2) }} {{ not not n }} {{ 1 if not no else 2 }}", "{{ n in [7] and not (n in [8]) }}", "{{ 'x' if 1 in xs and 2 in xs else 'y' }}",
    "{{ true and missing }}|{{ missing or 'z' }}|{{ missing and 1 }}|", "{{ missing + 1 }}", "{{ missing > 1 }}", "{{ missing == none }} {{ missing == missing }}", "{{ missing|upper }}|{{ missing|trim }}|",
    "{{ 'a'.split(1) }}", "{{ 'abc'.startswith('x') }} {{ 'abc'.endswith('c') }}", "{{ 'a'.replace(1, 2) }}", "{{ 'a'.replace('a') }}", "{{ 'a'.join() }}", "{{ 'a'.join(1) }}", "{{ 'x'.strip(1) }}", "{{ ' x '.strip(none) }}|{{ 'axa'.lstrip('a') }}|{{ 'axa'.rstrip('a') }}",
    "{{ people|join(',', 'name') }}", "{{ [1, 2]|join(attribute='x') }}|", "{{ nothing|first }}", "{{ n|first }}", "{{ n|list }}", "{{ n|reverse }}", "{{ n|sort }}", "{{ n|sum }}", "{{ ['a']|sum }}", 
    "{{ people|selectattr('age', 'nosuch')|list }}", "{{ people|selectattr('age', 'gt')|list }}", "{{ n|tojson(indent=none) }}", "{{ {'a': 1}|tojson(indent=0) }}", "{{ [1, [2, []]]|tojson(indent=2) }}",
    "{{ nothing|int }} {{ [1]|int }} {{ [1]|float }} {{ 'x'|float(2) }} {{ true|float }} {{ '1e2'|float }} {{ ' 7 '|int }}", "{{ 'a'|abs }}", "{{ n is sequence }} {{ d.get('a',) }} {{ range(2,)|list }}", "{% macro m(a,) %}{{ a }}{% endmacro %}{{ m(1,) }}", "{{ 5|int }} {{ true|int }} {{ [1, 2]|last }} {{ 'ab'|last }} {{ 'ab'|first }}", "{{ nothing|last }}", "{{ {}|first }}", "{{ xs|map(attribute='x', default='d')|list }}", "{{ [1]|map(attribute='x')|list|length }}",
    "{{ 'x' is odd }} {{ n is divisibleby }}", "{{ n is divisibleby 0 }}", "{{ n is divisibleby('a') }}", "{{ d.get is callable }} {{ n is callable }}", "{{ n is in }}", "{{ n is eq }}", "{{ missing is callable }}", "{{ nothing is string }} {{ 1.5 is number }} {{ 'a' is boolean }}",
    "{{ xs[none:] }} {{ xs[:none] }} {{ xs[none::none] }}", "{{ xs[1.5:] }}", "{{ n[1:] }}", "{{ missing[1:] }}", "{{ xs[::0] }}", "{{ n() }}", "{{ d.a() }}", "{{ -yes }} {{ -no }}", "{{ -'a' }}", "{{ +'a' }}", "{{ 1 // 0 }}", "{{ 1.5 % 0 }}", "{{ 1.5 // 0 }}", "{{ 1.5 / 0 }}",
    "{{ 'abc'[-1] }} {{ 'abc'[9] }}|{{ xs[-9] }}|{{ xs['a'] }}|{{ d[1] }}|{{ s['a'] }}", "{{ missing[0] }}", "{% set ns = namespace(a=1) %}{{ ns['a'] }}|{{ ns['b'] }}|{{ ns.a }}", "{% set ns = namespace(a=1) %}{{ 'a' in ns }}", "{{ n[0] }}", "{{ nothing['a'] }}",
    "{{ 'a\\rb\\bc\\fd\\ve\\af\\0g'|length }}", "x{{ 'p\\\nq' }}y", "{{ n|a.b }}", "{{ 'x'|upper|lower|title|capitalize }}", "{{ 'a'|string|safe }}", "{{ [1, 2, 3]|reverse|first }}",
    "{{ namespace(1) }}", "{{ namespace(d).a }}", "{{ dict(1) }}", "{{ dict(d, b=2) }}", "{{ range('a') }}", "{{ range(1, 2, 3, 4) }}", "{{ range() }}", "{{ range(1000000000)|length }}", "{{ range(5, 0)|list }}", "{{ range(0, 6, 2)|list }}",
    "{% for x in [[1, 2]] %}{% for y in x %}{{ loop.length }}{% endfor %}{% endfor %}", "{{ loop }}", 
    "{{ d.get.x }}", "{{ nothing.x.y }}",
    "{{ 5|string|length }}", "{{ 'a-b'.split('-')|first }}", "{{ d.get('b')|length }}", "{{ s|trim|upper|list|first }}", "{{ words|map('upper')|join('/') }}", "{{ [1, 2, 3]|map('string')|join }}",
]


def snippet_cases(env):
    cases = []
    for source in SNIPPETS:
        cases.append(dict(template=source, **render(env, source, CONTEXT)))
    return {"context": CONTEXT, "cases": cases}


def strftime_cases():
    import datetime
    moment = datetime.datetime(2024, 7, 5, 14, 3, 9)
    formats = ["%Y-%m-%d %H:%M:%S", "%a %A %b %B", "%e|%d|%j", "%I %p", "%y %%", "100%", "%Q"]
    return {"moment": moment.isoformat(), "cases": [{"format": f, "output": moment.strftime(f)} for f in formats]}


def main():
    out, args = sys.argv[1], sys.argv[2:]
    env = environment()
    result = {"models": [], "fixed_date": FIXED_DATE}
    for arg in args:
        if arg == "--snippets":
            result["snippets"] = snippet_cases(env)
            result["strftime"] = strftime_cases()
        else:
            name, path = arg.split("=", 1)
            result["models"].append(model_cases(env, name, path))
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"{len(result['models'])} models, {len(result.get('snippets', {}).get('cases', []))} snippets")


if __name__ == "__main__":
    main()
