
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

project = "Yapay Öğrenme Sözlüğü"
copyright = "2020, All authors. Licensed under CC-BY-SA-4.0 and MIT-0."
author = "The contributors"
release = "0.0.1"

extensions = ["recommonmark","sphinxcontrib.bibtex","sphinxcontrib.rsvgconverter","sphinx.ext.autodoc","sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

suppress_warnings = ['misc.highlighting_failure']
linkcheck_ignore = [r'.*localhost.*']
linkcheck_timeout = 5
linkcheck_workers = 20

autodoc_default_options = {
    'undoc-members': True,
    'show-inheritance': True,
}


html_theme = 'mxtheme'
html_theme_options = {
    'primary_color': 'red',
    'accent_color': 'deep_orange',
    'header_links': [
        
    ],
    'show_footer': False
}
html_static_path = ['_static']

html_favicon = ''

html_logo = ''

latex_documents = [
    (master_doc, "yapayogrenmesozlugu.tex", "Yapay Öğrenme Sözlüğü",
     author, 'manual'),
]

rsvg_converter_args = ['-z', '0.8']

latex_engine = 'xelatex' # for utf-8 supports
latex_show_pagerefs = True
latex_show_urls = 'footnote'

latex_logo = ''

latex_elements = {

'figure_align': 'H',

'pointsize': '11pt',
'preamble': r'''

% Page size
\setlength{\voffset}{-14mm}
\addtolength{\textheight}{16mm}

% Chapter title style
\usepackage{titlesec, blindtext, color}
\definecolor{gray75}{gray}{0.75}
\newcommand{\hsp}{\hspace{20pt}}
\titleformat{\chapter}[hang]{\Huge\bfseries}{\thechapter\hsp\textcolor{gray75}{|}\hsp}{0pt}{\Huge\bfseries}

% So some large pictures won't get the full page
\renewcommand{\floatpagefraction}{.8}

\setcounter{tocdepth}{1}
% Use natbib's citation style, e.g. (Li and Smola, 16)
\usepackage{natbib}
\protected\def\sphinxcite{\citep}





% Remove top header
\usepackage[draft]{minted}
\fvset{breaklines=true, breakanywhere=true}
\setlength{\headheight}{13.6pt}
\makeatletter
    \fancypagestyle{normal}{
        \fancyhf{}
        \fancyfoot[LE,RO]{{\py@HeaderFamily\thepage}}
        \fancyfoot[LO]{{\py@HeaderFamily\nouppercase{\rightmark}}}
        \fancyfoot[RE]{{\py@HeaderFamily\nouppercase{\leftmark}}}
        \fancyhead[LE,RO]{{\py@HeaderFamily }}
     }
\makeatother
''',
'sphinxsetup': 'verbatimwithframe=false, verbatimsep=2mm, VerbatimColor={rgb}{.95,.95,.95}'
}



def setup(app):
    # app.add_js_file('https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js')
    app.add_js_file('d2l.js')
    app.add_css_file('d2l.css')
    import mxtheme
    app.add_directive('card', mxtheme.CardDirective)
