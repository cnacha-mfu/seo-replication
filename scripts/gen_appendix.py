import json, datetime

with open('C:/Users/DELL/git/senso-framework/experiments/jcsse2026/results/expanded_corpus.json', encoding='utf-8') as f:
    corpus = json.load(f)

with open('C:/Users/DELL/git/senso-framework/experiments/jcsse2026/results/rq1_category_distribution.json', encoding='utf-8') as f:
    rq1 = json.load(f)

evo_lookup = {pm['repo']: pm['evolution_context_present'] for pm in rq1['per_manifest']}
today = datetime.date(2026, 3, 23)

TYPE_LABEL = {
    'claude_md':   'CLAUDE.md',
    'claude_code': 'CLAUDE.md',
    'codex':       'AGENTS.md',
    'copilot':     'copilot-instructions.md',
    'cline':       '.clinerules',
}
TYPE_ORDER = {'claude_md': 0, 'claude_code': 0, 'codex': 1, 'copilot': 2, 'cline': 3}

rows = []
for r in corpus:
    name = r['full_name']
    lang = (r.get('language') or 'N/A').capitalize()[:12]
    stars = r.get('stars', 0)
    created = r.get('created_at', '')[:10]
    age = (today - datetime.date.fromisoformat(created)).days // 30 if created else 0
    raw_type = r.get('manifest_type') or 'claude_md'
    label = TYPE_LABEL.get(raw_type, raw_type)
    evo = evo_lookup.get(name, False)
    rows.append((TYPE_ORDER.get(raw_type, 9), label, name, lang, stars, age, evo))

rows.sort(key=lambda x: (x[0], x[2].lower()))


def esc(s):
    return (s.replace('_', r'\_')
             .replace('#', r'\#')
             .replace('&', r'\&')
             .replace('%', r'\%'))


ABBREVS = {
    'derrickburns/generalized-kmeans-clustering': 'derrickburns/gen-kmeans-clust.',
    'javascript-obfuscator/javascript-obfuscator': 'js-obfuscator/js-obfuscator',
    'kubernetes-sigs/cloud-provider-azure':        'kubernetes-sigs/cloud-prov-az',
    'snowplow/snowplow-javascript-tracker':         'snowplow/snowplow-js-tracker',
    'hapifhir/org.hl7.fhir.validator-wrapper':      'hapifhir/hl7.fhir.valid.-wrap',
    'VCnoC/Claude-Code-Zen-mcp-Skill-Work':         'VCnoC/Claude-Code-Zen-mcp-SW',
    'HassanZahirnia/laravel-package-ocean':          'HassanZahirnia/laravel-pkg-oc',
    'microsoftgraph/msgraph-sdk-dotnet':             'msgraph/msgraph-sdk-dotnet',
    'popstas/telegram-download-chat':                'popstas/telegram-dl-chat',
    'daohainam/microservice-patterns':               'daohainam/microservice-patt.',
}


def format_repo(name):
    # Use \path{} which renders in monospace AND allows line breaks at / - .
    display = ABBREVS.get(name, name)
    return r'\path{' + display + '}'


def build_tabular_lines(data_rows, start_idx=1):
    """Build lines for one half-table (rows only, no header/footer)."""
    lines = []
    prev_label = None
    idx = start_idx
    for (_, label, name, lang, stars, age, evo) in data_rows:
        if label != prev_label:
            if prev_label is not None:
                lines.append(r'\midrule')
            # Group header as a non-counting row
            lines.append(
                r'\multicolumn{6}{l}{\textbf{\textit{' + label + r'}}} \\'
            )
            lines.append(r'\noalign{\vskip-2pt}\midrule')
            prev_label = label
        repo_cell = format_repo(name)
        stars_fmt = '{:,}'.format(stars)
        evo_cell = r'$\checkmark$' if evo else '---'
        lines.append(
            str(idx) + ' & ' + repo_cell + ' & ' + esc(lang) + ' & ' +
            stars_fmt + ' & ' + str(age) + ' & ' + evo_cell + r' \\'
        )
        idx += 1
    return '\n'.join(lines)


left_content = build_tabular_lines(rows[:40], start_idx=1)
right_content = build_tabular_lines(rows[40:], start_idx=41)

# Write complete appendix block to embed directly in .tex
appendix_block = r"""% ---------------------------------------------------------------
\appendix
\section{Complete Corpus Listing}
\label{app:corpus}

Table~\ref{tab:corpus} lists all 80 repositories, sorted by manifest
type then alphabetically.
Columns: sequential index, repository (\texttt{owner/repo}), primary
language, GitHub star count, age in months at collection (Mar 2026),
and evolution context detected ($\checkmark$~= yes, ---~= no).
Names exceeding 29 characters are abbreviated.

\begin{table*}[ht]
\caption{Complete corpus of 80 repositories.
Grouped by manifest type; sorted alphabetically within each group.
$\checkmark$~= evolution context detected; ---~= absent.}
\label{tab:corpus}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
%
\begin{minipage}[t]{0.485\textwidth}
\begin{tabular}{rp{3.6cm}lrrc}
\toprule
\# & Repository & Lang. & Stars & Age & E \\
\midrule
""" + left_content + r"""
\bottomrule
\end{tabular}
\end{minipage}%
\hfill%
\begin{minipage}[t]{0.485\textwidth}
\begin{tabular}{rp{3.6cm}lrrc}
\toprule
\# & Repository & Lang. & Stars & Age & E \\
\midrule
""" + right_content + r"""
\bottomrule
\end{tabular}
\end{minipage}
\end{table*}"""

with open('C:/Users/DELL/git/senso-framework/papers/jcsse2026/appendix_block.tex', 'w', encoding='utf-8') as f:
    f.write(appendix_block)

print('Appendix block written.')
print(f'Left rows: {len(rows[:40])}, Right rows: {len(rows[40:])}')
