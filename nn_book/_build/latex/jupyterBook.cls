\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{jupyterBook}[2020/11/06]

\RequirePackage{xcolor}
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage{amsmath, amssymb}
\RequirePackage{mathrsfs}
\RequirePackage{tikz}
\usetikzlibrary{arrows}
\usetikzlibrary{calc}
\usetikzlibrary{intersections}
\usetikzlibrary{decorations}
\RequirePackage{pgf}
\RequirePackage{pgfplots}
\RequirePackage{bbm}

\def\sphinxdocclass{report}
\LoadClass{sphinxmanual}

\setlength{\textwidth}{17.5cm}
\setlength{\textheight}{22cm}

\newcommand{\RR}{\mathbbm R}
\newcommand{\NN}{\mathbbm N}
\newcommand{\PP}{\mathbbm P}
\newcommand{\EE}{\mathbbm E \,}
\newcommand{\XX}{\mathbbm X}
\newcommand{\ZZ}{\mathbbm Z}
\newcommand{\QQ}{\mathbbm Q}
\newcommand{\fF}{\mathcal F}
\newcommand{\dD}{\mathcal D}
\newcommand{\lL}{\mathcal L}
\newcommand{\gG}{\mathcal G}
\newcommand{\hH}{\mathcal H}
\newcommand{\nN}{\mathcal N}
\newcommand{\pP}{\mathcal P}
\DeclareMathOperator{\trace}{trace}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Span}{span}
\DeclareMathOperator{\proj}{proj}
\DeclareMathOperator{\col}{col}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\gt}{>}
