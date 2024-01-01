\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ece459}[2023-09-14 ECE 459 Final Report Template]

\LoadClass[a4paper,10pt,twoside]{book}

\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage[format=hang,font={small},labelfont={bf},labelsep={quad}]{caption}
\RequirePackage{amsmath, amssymb, amsfonts}  % for math spacing
\RequirePackage[margin=1in]{geometry} % page layout 1' margin
\RequirePackage{enumitem}
\RequirePackage{verbatim} % allows for verbatim text ignoring any TeX commands inside "verbatim" environment
\RequirePackage{listings}
\RequirePackage[dvipsnames,svgnames,table,xcdraw]{xcolor}
\RequirePackage[ruled,hangingcomment,nofillcomment,noend]{algorithm2e}% pseudo-code algorithm support
\RequirePackage[bottom]{footmisc}
\RequirePackage{float}
\RequirePackage{anyfontsize}
\RequirePackage{csquotes} % Provide displayquote
\RequirePackage[style=ieee]{biblatex}
\RequirePackage[titletoc]{appendix}
\renewcommand{\appendixname}{Appendix}
\renewbibmacro*{date}{%
  \iffieldundef{year}
  {\bibstring{nodate}}
  {\printdate}}

\RequirePackage{fontspec}
\RequirePackage[scheme=plain, fontset=fandol]{ctex}\defaultfontfeatures{}
\RequirePackage{titlesec}
\RequirePackage{titletoc}
\RequirePackage{hyperref}
\RequirePackage{xurl}
\RequirePackage{setspace}
\RequirePackage{afterpage}
\RequirePackage[per-mode=symbol,inter-unit-product = \ensuremath{{}\cdot{}},mode=math]{siunitx}
\DeclareSIPrefix\micro{\text{\textmu}}{-3} % typing SI Units, not necessary for default LMC font
\onehalfspacing

% Font settings
\renewcommand*{\UrlFont}{\rmfamily}
\newfontfamily{\cambria}{Cambria}
\newfontfamily{\calibri}{Calibri}
\everymath{\displaystyle}

% Chapter/section/subsection title format
\titleformat{\chapter}
{\fontsize{16pt}{\baselineskip}\selectfont\bfseries}
{\thechapter}{1em}{}
\titlespacing{\chapter}{0pt}{0pt}{\baselineskip}

\titleformat{\section}
{\fontsize{14pt}{\baselineskip}\selectfont\bfseries}
{\thesection}{1em}{}

\titleformat{\subsection}
{\fontsize{12pt}{\baselineskip}\selectfont\bfseries}
{\thesubsection}{1em}{}

\titleformat{\subsubsection}
{\fontsize{11pt}{\baselineskip}\selectfont\bfseries}
{\thesubsubsection}{1em}{}

% Enumerate list format
\setenumerate[1]{
  itemsep=0pt,partopsep=0pt,parsep=0pt,topsep=2ex}
\setitemize[1]{
  itemsep=0pt,partopsep=0pt,parsep=0pt,topsep=2ex}
\setdescription{
  itemsep=0pt,partopsep=0pt,parsep=0pt,topsep=2ex}
\setlength{\parindent}{2em}
\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
% Header and footer
\fancypagestyle{plain}{
  \fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
  \fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont\leftmark}
  \fancyfoot[C]{\thepage}
}
\fancypagestyle{toc}{
  \fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
  \fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont Contents}
  \fancyfoot[C]{\thepage}
}
\fancypagestyle{acknowledgement}{
  \fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
  \fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont Acknolwedgement}
  \fancyfoot[C]{\thepage}
}
\fancypagestyle{soc}{
  \fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
  \fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont Statement of Contribution}
  \fancyfoot[C]{\thepage}
}
\fancypagestyle{abstract}{
  \fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
  \fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont Abstract}
  \fancyfoot[C]{\thepage}
}
\fancypagestyle{references}{
  \fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
  \fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont References}
  \fancyfoot[C]{\thepage}
}
\fancyhead[L]{\fontsize{9pt}{\baselineskip}\selectfont\MakeCapital{\RPTTYPE}}
\fancyhead[R]{\fontsize{9pt}{\baselineskip}\selectfont\leftmark}
\fancyfoot[C]{\thepage}

\lstset{
  basicstyle=\small\ttfamily,
  numbers=left,
  numberstyle=\scriptsize,
  tabsize=4,
  breaklines,
  columns=fixed,
  keepspaces,
  commentstyle=\itshape\color{ForestGreen},
  keywordstyle=\color{Blue}\bfseries,
}
\newcommand\reporttitle[1]{\def\RPTTITLE{#1}}
\newcommand\reportdate[1]{\def\RPTDATE{#1}}
\newcommand\authornames[1]{\def\AUTHORNAME{#1}}
\newcommand\instructor[1]{\def\FACULTYNAME{#1}}
\newcommand{\HRule}{\rule{\linewidth}{0.1mm}}
\newcommand\reporttype[1]{\def\RPTTYPE{#1}}
\newcommand\coursecode[1]{\def\COURSECODE{#1}}
\newcommand\coursename[1]{\def\COURSENAME{#1}}
\newcommand\logo[1]{\def\LOGOPATH{#1}}
\newcommand\name[2]{#1~\textsc{#2}}

\def\RPTTYPE{~}
\def\RPTTITLE{~}
\def\RPTDATE{~}
\def\AUTHORNAME{~}
\def\FACULTYNAME{~}
\def\COURSECODE{~}
\def\COURSENAME{~}


\newcommand\makecoverpage{
  \begin{titlepage}
    \center % Center everything on the page
    \sf
    \includegraphics[width=0.6\linewidth]{\LOGOPATH}\\[1cm]
    \textsc{\Large\COURSECODE}\\[0.5cm] % heading course Number
    \textsc{\Large\COURSENAME}\\[0.5cm] % heading course name
    {\Large\RPTTYPE}\\[1cm] % Minor heading
    \HRule \\[0.5cm]
    \huge \textbf{\MakeUppercase{\RPTTITLE}} % Title
    \HRule \\
    \vfill
    {\large\AUTHORNAME}
    \vfill
    {\large{\underline{\textbf{Instructor}}}\\\FACULTYNAME}\\
    \vfill
    {\large \RPTDATE} % Date

  \end{titlepage}
}
