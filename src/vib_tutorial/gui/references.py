"""The books and papers the notes point to, in one list: each page's *Further reading* is made from it.

A notes file cites a reference in its text with ``<cite>key</cite>`` (written out as a link, "Den Hartog
1956", to the entry below) and lists more to read with ``<reading>key key …</reading>`` (anywhere; it shows
nothing where it stands). `links.load_notes` ends the page with a *Further reading* section: what it cites,
in the order it first does, then the rest of its reading.

Working on it: `tests/test_notation.py` checks every key the notes use is here and every entry here is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    key: str
    cite: str  # how the text names it: "Den Hartog 1956"
    entry: str  # HTML: authors, title (in <i>), edition, publisher, year
    note: str = ""  # what it's good for, a few words


REFERENCES = (
    Reference(
        "rao",
        "Rao 2017",
        "S. S. Rao, <i>Mechanical Vibrations</i>, 6th edition (SI), Pearson, 2017",
        "the broadest of the textbooks, with many worked examples",
    ),
    Reference(
        "inman",
        "Inman 2014",
        "D. J. Inman, <i>Engineering Vibration</i>, 4th edition, Pearson, 2014",
        "clear on one mass, modal analysis and measurement",
    ),
    Reference(
        "thomson",
        "Thomson and Dahleh 1998",
        "W. T. Thomson and M. D. Dahleh, <i>Theory of Vibration with Applications</i>, 5th edition, "
        "Prentice Hall, 1998",
        "compact, with the classical results derived",
    ),
    Reference(
        "den-hartog",
        "Den Hartog 1956",
        "J. P. Den Hartog, <i>Mechanical Vibrations</i>, 4th edition, McGraw-Hill, 1956 (Dover reprint, 1985)",
        "the classic: physical explanations, absorbers, rotors",
    ),
    Reference(
        "den-hartog-1931",
        "Den Hartog 1931",
        "J. P. Den Hartog, Forced vibrations with combined Coulomb and viscous friction, <i>Transactions of the "
        "ASME</i> 53 (APM-53-9), 107–115, 1931",
        "the exact sliding motion, and when friction bounds a resonance",
    ),
    Reference(
        "coulomb",
        "Coulomb 1785",
        "C.-A. Coulomb, Théorie des machines simples, en ayant égard au frottement de leurs parties et à la roideur "
        "des cordages, <i>Mémoires de mathématique et de physique présentés à l'Académie royale des sciences</i> "
        "10, 1785 (the Academy's prize of 1781)",
        "dry friction's laws, from experiment",
    ),
    Reference(
        "crandall",
        "Crandall 1970",
        "S. H. Crandall, The role of damping in vibration theory, <i>Journal of Sound and Vibration</i> 11, 3–18, 1970",
        "what each damping model can and can't do, hysteretic damping's lack of a causal form among them",
    ),
    Reference(
        "maxwell",
        "Maxwell 1867",
        "J. C. Maxwell, On the dynamical theory of gases, <i>Philosophical Transactions of the Royal Society of "
        "London</i> 157, 49–88, 1867",
        "the model of a stress that relaxes, a spring and a viscosity in series",
    ),
    Reference(
        "nashif",
        "Nashif, Jones and Henderson 1985",
        "A. D. Nashif, D. I. G. Jones and J. P. Henderson, <i>Vibration Damping</i>, Wiley, 1985",
        "damping materials, their measurement and their use in structures",
    ),
    Reference(
        "rivin",
        "Rivin 2003",
        "E. I. Rivin, <i>Passive Vibration Isolation</i>, ASME Press, 2003",
        "isolation in practice: six degrees of freedom, two stages, nonlinear mounts, waves in isolators",
    ),
    Reference(
        "zener",
        "Zener 1948",
        "C. Zener, <i>Elasticity and Anelasticity of Metals</i>, University of Chicago Press, 1948",
        "the standard linear solid and its relaxation peak",
    ),
    Reference(
        "baker-blackburn",
        "Baker and Blackburn 2005",
        "G. L. Baker and J. A. Blackburn, <i>The Pendulum: A Case Study in Physics</i>, Oxford University Press, 2005",
        "the pendulum in full: large swings, the physical and torsion pendulums, Kater's, chaos",
    ),
    Reference(
        "duffing",
        "Duffing 1918",
        "G. Duffing, <i>Erzwungene Schwingungen bei veränderlicher Eigenfrequenz und ihre technische "
        "Bedeutung</i>, Sammlung Vieweg, Heft 41/42, Vieweg, Braunschweig, 1918",
        "the oscillator with a cubic spring",
    ),
    Reference(
        "kater",
        "Kater 1818",
        "H. Kater, An account of experiments for determining the length of the pendulum vibrating seconds in "
        "the latitude of London, <i>Philosophical Transactions of the Royal Society of London</i> 108, 33–102, "
        "1818",
        "the reversible pendulum",
    ),
    Reference(
        "duhamel",
        "Duhamel 1833",
        "J.-M. C. Duhamel, Sur la méthode générale relative au mouvement de la chaleur dans les corps solides "
        "plongés dans des milieux dont la température varie avec le temps, <i>Journal de l'École polytechnique</i> "
        "14 (22e cahier), 20–77, 1833",
        "the superposition of responses to a changing input, first for heat",
    ),
    Reference(
        "harris",
        "Piersol and Paez 2010",
        "A. G. Piersol and T. L. Paez (eds), <i>Harris' Shock and Vibration Handbook</i>, 6th edition, "
        "McGraw-Hill, 2010",
        "the engineer's reference: shock pulses and spectra, isolators and six-degree-of-freedom mountings",
    ),
    Reference(
        "ogata",
        "Ogata 2010",
        "K. Ogata, <i>Modern Control Engineering</i>, 5th edition, Prentice Hall, 2010",
        "the Laplace transform and transfer functions, step responses' overshoot and settling",
    ),
    Reference(
        "fourier",
        "Fourier 1822",
        "J. Fourier, <i>Théorie analytique de la chaleur</i>, Firmin Didot, Paris, 1822",
        "where the series were first put to work, on heat flow",
    ),
    Reference(
        "gibbs",
        "Gibbs 1899",
        "J. W. Gibbs, Fourier's series, <i>Nature</i> 59, 200 (1898) and 59, 606 (1899)",
        "two letters; the second describes the overshoot",
    ),
    Reference(
        "hewitt",
        "Hewitt and Hewitt 1979",
        "E. Hewitt and R. E. Hewitt, The Gibbs–Wilbraham phenomenon: an episode in Fourier analysis, "
        "<i>Archive for History of Exact Sciences</i> 21, 129–160, 1979",
        "the overshoot's history, from Wilbraham to Gibbs and after",
    ),
    Reference(
        "kreyszig",
        "Kreyszig 2011",
        "E. Kreyszig, <i>Advanced Engineering Mathematics</i>, 10th edition, Wiley, 2011",
        "Fourier series from the start, with their tables and symmetries",
    ),
    Reference(
        "parseval",
        "Parseval 1806",
        "M.-A. Parseval des Chênes, Mémoire sur les séries et sur l'intégration complète d'une équation aux "
        "différences partielles linéaires du second ordre, à coefficients constants, <i>Mémoires présentés à "
        "l'Institut des Sciences, Lettres et Arts, par divers savans</i> 1, 1806 (read 5 April 1799)",
        "",
    ),
    Reference(
        "wilbraham",
        "Wilbraham 1848",
        "H. Wilbraham, On a certain periodic function, <i>Cambridge and Dublin Mathematical Journal</i> 3, "
        "198–201, 1848",
        "the overshoot, fifty years before Gibbs",
    ),
    Reference(
        "lagrange",
        "Lagrange 1788",
        "J.-L. Lagrange, <i>Méchanique analitique</i>, Veuve Desaint, Paris, 1788",
        "mechanics from the energies, without a figure in it",
    ),
    Reference(
        "rayleigh",
        "Rayleigh 1873",
        "J. W. Strutt (Lord Rayleigh), Some general theorems relating to vibrations, <i>Proceedings of the London "
        "Mathematical Society</i> s1-4, 357–368, 1873",
        "where the dissipation function comes in",
    ),
    Reference(
        "milliken",
        "Milliken and Milliken 2002",
        "W. F. Milliken and D. L. Milliken, <i>Chassis Design: Principles and Analysis</i> (from Maurice Olley's "
        "notes), SAE International, 2002",
        "Olley's own ride work, flat ride among it",
    ),
    Reference(
        "gillespie",
        "Gillespie 2021",
        "T. D. Gillespie, <i>Fundamentals of Vehicle Dynamics</i>, revised edition, SAE International, 2021",
        "a car's ride, bounce and pitch",
    ),
    Reference(
        "shinbrot",
        "Shinbrot et al. 1992",
        "T. Shinbrot, C. Grebogi, J. Wisdom and J. A. Yorke, Chaos in a double pendulum, <i>American Journal of "
        "Physics</i> 60, 491–499, 1992",
        "",
    ),
    Reference(
        "rayleigh-sound",
        "Rayleigh 1877",
        "J. W. Strutt (Lord Rayleigh), <i>The Theory of Sound</i>, vol. 1, Macmillan, 1877 (2nd edition 1894)",
        "the quotient and proportional damping, among much else",
    ),
    Reference(
        "ritz",
        "Ritz 1909",
        "W. Ritz, Über eine neue Methode zur Lösung gewisser Variationsprobleme der mathematischen Physik, "
        "<i>Journal für die reine und angewandte Mathematik</i> 135, 1–61, 1909",
        "",
    ),
    Reference(
        "caughey",
        "Caughey and O'Kelly 1965",
        "T. K. Caughey and M. E. J. O'Kelly, Classical normal modes in damped linear dynamic systems, <i>Journal "
        "of Applied Mechanics</i> 32, 583–588, 1965",
        "",
    ),
    Reference(
        "frahm",
        "Frahm 1911",
        "H. Frahm, Device for damping vibrations of bodies, US patent 989,958, 1911",
        "",
    ),
    Reference(
        "ormondroyd",
        "Ormondroyd and Den Hartog 1928",
        "J. Ormondroyd and J. P. Den Hartog, The theory of the dynamic vibration absorber, <i>Transactions of the "
        "ASME</i> 50, 9–22, 1928",
        "",
    ),
    Reference(
        "brock",
        "Brock 1946",
        "J. E. Brock, A note on the damped vibration absorber, <i>Journal of Applied Mechanics</i> 13, A-284, 1946",
        "",
    ),
    Reference(
        "dunkerley",
        "Dunkerley 1894",
        "S. Dunkerley, On the whirling and vibration of shafts, <i>Philosophical Transactions of the Royal Society "
        "A</i> 185, 279–360, 1894",
        "",
    ),
    Reference(
        "holzer",
        "Holzer 1921",
        "H. Holzer, <i>Die Berechnung der Drehschwingungen und ihre Anwendung im Maschinenbau</i>, Springer, Berlin, "
        "1921",
        "",
    ),
    Reference(
        "poincare",
        "Poincaré 1890",
        "H. Poincaré, Sur les équations aux dérivées partielles de la physique mathématique, <i>American Journal "
        "of Mathematics</i> 12, 211–294, 1890",
        "",
    ),
    Reference(
        "wilson-ritz",
        "Wilson, Yuan and Dickens 1982",
        "E. L. Wilson, M.-W. Yuan and J. M. Dickens, Dynamic analysis by direct superposition of Ritz vectors, "
        "<i>Earthquake Engineering and Structural Dynamics</i> 10, 813–821, 1982",
        "",
    ),
    Reference(
        "maxwell-1864",
        "Maxwell 1864",
        "J. C. Maxwell, On the calculation of the equilibrium and stiffness of frames, <i>Philosophical Magazine</i> "
        "27, 294–299, 1864",
        "",
    ),
    Reference(
        "betti",
        "Betti 1872",
        "E. Betti, Teoria della elasticità, <i>Il Nuovo Cimento</i>, series 2, 7–8, 1872",
        "",
    ),
    Reference(
        "kennedy-pancu",
        "Kennedy and Pancu 1947",
        "C. C. Kennedy and C. D. P. Pancu, Use of vectors in vibration measurement and analysis, <i>Journal of the "
        "Aeronautical Sciences</i> 14, 603–625, 1947",
        "",
    ),
    Reference(
        "ewins",
        "Ewins 2000",
        "D. J. Ewins, <i>Modal Testing: Theory, Practice and Application</i>, 2nd edition, Research Studies Press, "
        "2000",
        "FRFs, their modal forms and residuals, and how they are measured: the standard text on modal testing",
    ),
    Reference(
        "foss",
        "Foss 1958",
        "K. A. Foss, Co-ordinates which uncouple the equations of motion of damped linear dynamic systems, "
        "<i>Journal of Applied Mechanics</i> 25, 361–364, 1958",
        "",
    ),
    Reference(
        "pappa",
        "Pappa, Elliott and Schenk 1993",
        "R. S. Pappa, K. B. Elliott and A. Schenk, Consistent-mode indicator for the eigensystem realization "
        "algorithm, <i>Journal of Guidance, Control, and Dynamics</i> 16, 852–858, 1993",
        "",
    ),
    Reference(
        "allemang-brown",
        "Allemang and Brown 1982",
        "R. J. Allemang and D. L. Brown, A correlation coefficient for modal vector analysis, <i>Proceedings of the "
        "1st International Modal Analysis Conference</i>, Orlando, 110–116, 1982",
        "",
    ),
    Reference(
        "adhikari",
        "Adhikari 2013",
        "S. Adhikari, <i>Structural Dynamic Analysis with Generalized Damping Models: Analysis</i>, Wiley-ISTE, 2013",
        "non-proportional damping and complex modes in depth: their perturbation forms, state space and beyond",
    ),
    Reference(
        "dalembert",
        "d'Alembert 1747",
        "J. le R. d'Alembert, Recherches sur la courbe que forme une corde tenduë mise en vibration, <i>Histoire de "
        "l'Académie royale des sciences et belles lettres de Berlin</i> 3, 214–219, 1747 (published 1749)",
        "",
    ),
    Reference(
        "euler-1744",
        "Euler 1744",
        "L. Euler, De curvis elasticis, Additamentum I to <i>Methodus inveniendi lineas curvas maximi minimive "
        "proprietate gaudentes</i>, Lausanne and Geneva, 1744",
        "",
    ),
    Reference(
        "brillouin",
        "Brillouin 1946",
        "L. Brillouin, <i>Wave Propagation in Periodic Structures</i>, McGraw-Hill, 1946 (2nd edition, Dover, 1953)",
        "",
    ),
    Reference(
        "archer",
        "Archer 1963",
        "J. S. Archer, Consistent mass matrix for distributed mass systems, <i>Journal of the Structural Division "
        "(ASCE)</i> 89, 161–178, 1963",
        "",
    ),
    Reference(
        "graff",
        "Graff 1975",
        "K. F. Graff, <i>Wave Motion in Elastic Solids</i>, 1975 (Dover reprint, 1991)",
        "waves in strings, bars, beams and plates: reflection, impedance and dispersion, worked in full",
    ),
    Reference(
        "rao-continuous",
        "Rao 2019",
        "S. S. Rao, <i>Vibration of Continuous Systems</i>, 2nd edition, Wiley, 2019",
        "the modes of strings, bars, beams and plates, with their end conditions, and their finite elements",
    ),
    Reference(
        "timoshenko",
        "Timoshenko 1921",
        "S. P. Timoshenko, On the correction for shear of the differential equation for transverse vibrations of "
        "prismatic bars, <i>Philosophical Magazine</i> (6) 41, 744–746, 1921",
        "",
    ),
    Reference(
        "kirchhoff",
        "Kirchhoff 1850",
        "G. Kirchhoff, Über das Gleichgewicht und die Bewegung einer elastischen Scheibe, <i>Journal für die reine "
        "und angewandte Mathematik</i> 40, 51–88, 1850",
        "",
    ),
    Reference(
        "chladni",
        "Chladni 1787",
        "E. F. F. Chladni, <i>Entdeckungen über die Theorie des Klanges</i>, Weidmanns Erben und Reich, Leipzig, 1787",
        "",
    ),
    Reference(
        "leissa",
        "Leissa 1969",
        "A. W. Leissa, <i>Vibration of Plates</i>, NASA SP-160, 1969 (reprinted by the Acoustical Society of "
        "America, 1993)",
        "the plates' frequencies tabulated, for every shape and edge: where the clamped circle's roots come from",
    ),
    Reference(
        "cfl",
        "Courant, Friedrichs and Lewy 1928",
        "R. Courant, K. Friedrichs and H. Lewy, Über die partiellen Differenzengleichungen der mathematischen Physik, "
        "<i>Mathematische Annalen</i> 100, 32–74, 1928",
        "",
    ),
    Reference(
        "newmark",
        "Newmark 1959",
        "N. M. Newmark, A method of computation for structural dynamics, <i>Journal of the Engineering Mechanics "
        "Division (ASCE)</i> 85, 67–94, 1959",
        "",
    ),
    Reference(
        "hht",
        "Hilber, Hughes and Taylor 1977",
        "H. M. Hilber, T. J. R. Hughes and R. L. Taylor, Improved numerical dissipation for time integration "
        "algorithms in structural dynamics, <i>Earthquake Engineering and Structural Dynamics</i> 5, 283–292, 1977",
        "",
    ),
    Reference(
        "runge",
        "Runge 1895",
        "C. Runge, Über die numerische Auflösung von Differentialgleichungen, <i>Mathematische Annalen</i> 46, "
        "167–178, 1895",
        "",
    ),
    Reference(
        "kutta",
        "Kutta 1901",
        "W. Kutta, Beitrag zur näherungsweisen Integration totaler Differentialgleichungen, <i>Zeitschrift für "
        "Mathematik und Physik</i> 46, 435–453, 1901",
        "",
    ),
    Reference(
        "hughes",
        "Hughes 2000",
        "T. J. R. Hughes, <i>The Finite Element Method: Linear Static and Dynamic Finite Element Analysis</i>, Dover, "
        "2000 (first published 1987)",
        "its chapters on time stepping: the amplification matrix, Newmark's family and HHT-α worked in full",
    ),
    Reference(
        "geradin-rixen",
        "Géradin and Rixen 2015",
        "M. Géradin and D. J. Rixen, <i>Mechanical Vibrations: Theory and Application to Structural Dynamics</i>, "
        "3rd edition, Wiley, 2015",
        "structural dynamics with the numerical side: time integration and its accuracy, among much else",
    ),
    Reference(
        "miles",
        "Miles 1954",
        "J. W. Miles, On structural fatigue under random loading, <i>Journal of the Aeronautical Sciences</i> 21, "
        "753–762, 1954",
        "",
    ),
    Reference(
        "rice",
        "Rice 1944–45",
        "S. O. Rice, Mathematical analysis of random noise, <i>Bell System Technical Journal</i> 23, 282–332, 1944, "
        "and 24, 46–156, 1945",
        "",
    ),
    Reference(
        "davenport",
        "Davenport 1964",
        "A. G. Davenport, Note on the distribution of the largest value of a random function with application to gust "
        "loading, <i>Proceedings of the Institution of Civil Engineers</i> 28, 187–196, 1964",
        "",
    ),
    Reference(
        "vanmarcke",
        "Vanmarcke 1975",
        "E. H. Vanmarcke, On the distribution of the first-passage time for normal stationary random processes, "
        "<i>Journal of Applied Mechanics</i> 42, 215–220, 1975",
        "",
    ),
    Reference(
        "welch",
        "Welch 1967",
        "P. D. Welch, The use of fast Fourier transform for the estimation of power spectra: a method based on time "
        "averaging over short, modified periodograms, <i>IEEE Transactions on Audio and Electroacoustics</i> 15, "
        "70–73, 1967",
        "",
    ),
    Reference(
        "scharton",
        "Scharton 1997",
        "T. D. Scharton, <i>Force Limited Vibration Testing Monograph</i>, NASA RP-1403, 1997",
        "",
    ),
    Reference(
        "bendat-piersol",
        "Bendat and Piersol 2010",
        "J. S. Bendat and A. G. Piersol, <i>Random Data: Analysis and Measurement Procedures</i>, 4th edition, Wiley, "
        "2010",
        "spectra, their estimates and their errors: the bias and random error of Welch's method in full",
    ),
    Reference(
        "newland",
        "Newland 1993",
        "D. E. Newland, <i>An Introduction to Random Vibrations, Spectral and Wavelet Analysis</i>, 3rd edition, "
        "Longman, 1993 (Dover reprint, 2005)",
        "random vibration from the ground up: spectra, the response of linear systems, crossings and peaks",
    ),
    Reference(
        "basquin",
        "Basquin 1910",
        "O. H. Basquin, The exponential law of endurance tests, <i>Proceedings of the American Society for Testing "
        "Materials</i> 10, 625–630, 1910",
        "",
    ),
    Reference(
        "palmgren",
        "Palmgren 1924",
        "A. Palmgren, Die Lebensdauer von Kugellagern, <i>Zeitschrift des Vereins Deutscher Ingenieure</i> 68, "
        "339–341, 1924",
        "",
    ),
    Reference(
        "miner",
        "Miner 1945",
        "M. A. Miner, Cumulative damage in fatigue, <i>Journal of Applied Mechanics</i> 12, A159–A164, 1945",
        "",
    ),
    Reference(
        "goodman",
        "Goodman 1899",
        "J. Goodman, <i>Mechanics Applied to Engineering</i>, Longmans, Green, London, 1899",
        "",
    ),
    Reference(
        "gerber",
        "Gerber 1874",
        "W. Gerber, Bestimmung der zulässigen Spannungen in Eisenconstructionen, <i>Zeitschrift des Bayerischen "
        "Architekten- und Ingenieur-Vereins</i> 6, 101–110, 1874",
        "",
    ),
    Reference(
        "en1993-1-9",
        "EN 1993-1-9",
        "EN 1993-1-9, <i>Eurocode 3: Design of steel structures, Part 1-9: Fatigue</i>, CEN, 2005",
        "",
    ),
    Reference(
        "peterson",
        "Pilkey and Pilkey 2008",
        "W. D. Pilkey and D. F. Pilkey, <i>Peterson's Stress Concentration Factors</i>, 3rd edition, Wiley, 2008",
        "",
    ),
    Reference(
        "matsuishi-endo",
        "Matsuishi and Endo 1968",
        "M. Matsuishi and T. Endo, Fatigue of metals subjected to varying stress, Japan Society of Mechanical "
        "Engineers, 1968",
        "",
    ),
    Reference(
        "astm-e1049",
        "ASTM E1049",
        "ASTM E1049-85, <i>Standard Practices for Cycle Counting in Fatigue Analysis</i>, ASTM International, 1985 "
        "(reapproved since)",
        "",
    ),
    Reference(
        "bendat-1964",
        "Bendat 1964",
        "J. S. Bendat, <i>Probability Functions for Random Responses: Prediction of Peaks, Fatigue Damage, and "
        "Catastrophic Failures</i>, NASA CR-33, 1964",
        "",
    ),
    Reference(
        "dirlik",
        "Dirlik 1985",
        "T. Dirlik, <i>Application of Computers in Fatigue Analysis</i>, PhD thesis, University of Warwick, 1985",
        "",
    ),
    Reference(
        "lalanne-4",
        "Lalanne 2014",
        "C. Lalanne, <i>Mechanical Vibration and Shock Analysis, Volume 4: Fatigue Damage</i>, 3rd edition, "
        "Wiley-ISTE, 2014",
        "fatigue under vibration from the S–N curve to the fatigue damage spectrum, with every spectral method",
    ),
    Reference(
        "schijve",
        "Schijve 2009",
        "J. Schijve, <i>Fatigue of Structures and Materials</i>, 2nd edition, Springer, 2009",
        "the materials side: crack initiation and growth, notches, mean stress and what the rules leave out",
    ),
    Reference(
        "iso8608",
        "ISO 8608:2016",
        "ISO 8608:2016, <i>Mechanical vibration: Road surface profiles: Reporting of measured data</i>, ISO, 2016",
        "",
    ),
    Reference(
        "iso2631",
        "ISO 2631-1:1997",
        "ISO 2631-1:1997, <i>Mechanical vibration and shock: Evaluation of human exposure to whole-body vibration, "
        "Part 1: General requirements</i>, ISO, 1997",
        "",
    ),
    Reference(
        "griffin",
        "Griffin 1990",
        "M. J. Griffin, <i>Handbook of Human Vibration</i>, Academic Press, 1990",
        "",
    ),
    Reference(
        "wong-vehicles",
        "Wong 2022",
        "J. Y. Wong, <i>Theory of Ground Vehicles</i>, 5th edition, Wiley, 2022",
        "ride and handling from the tyre up: road spectra, quarter- and half-car models, suspension design",
    ),
    Reference(
        "jeong-singh",
        "Jeong and Singh 2000",
        "T. Jeong and R. Singh, Analytical methods of decoupling the automotive engine torque roll axis, <i>Journal of "
        "Sound and Vibration</i> 234, 85–114, 2000",
        "",
    ),
    Reference(
        "biot",
        "Biot 1932",
        "M. A. Biot, <i>Vibration of Buildings during Earthquakes</i>, PhD thesis, California Institute of "
        "Technology, 1932; and A mechanical analyzer for the prediction of earthquake stresses, <i>Bulletin of the "
        "Seismological Society of America</i> 31, 151–171, 1941",
        "",
    ),
    Reference(
        "ohara-belsheim",
        "O'Hara and Belsheim 1963",
        "G. J. O'Hara and R. O. Belsheim, <i>Interim Design Values for Shock Design of Shipboard Equipment</i>, NRL "
        "Memorandum Report 1396, Naval Research Laboratory, 1963",
        "",
    ),
    Reference(
        "lalanne-2",
        "Lalanne 2014",
        "C. Lalanne, <i>Mechanical Vibration and Shock Analysis, Volume 2: Mechanical Shock</i>, 3rd edition, "
        "Wiley-ISTE, 2014",
        "the shock response spectrum, its properties, and how shock tests are specified and run on machines and shakers",
    ),
    Reference(
        "en1998-1",
        "EN 1998-1",
        "EN 1998-1, <i>Eurocode 8: Design of structures for earthquake resistance, Part 1: General rules, seismic "
        "actions and rules for buildings</i>, CEN, 2004",
        "",
    ),
    Reference(
        "der-kiureghian",
        "Der Kiureghian 1981",
        "A. Der Kiureghian, A response spectrum method for random vibration analysis of MDF systems, "
        "<i>Earthquake Engineering and Structural Dynamics</i> 9, 419–435, 1981",
        "",
    ),
    Reference(
        "veletsos-newmark",
        "Veletsos and Newmark 1960",
        "A. S. Veletsos and N. M. Newmark, Effect of inelastic behavior on the response of simple systems to "
        "earthquake motions, <i>Proceedings of the 2nd World Conference on Earthquake Engineering</i>, Tokyo, vol. 2, "
        "1960",
        "",
    ),
    Reference(
        "naeim-kelly",
        "Naeim and Kelly 1999",
        "F. Naeim and J. M. Kelly, <i>Design of Seismic Isolated Structures: From Theory to Practice</i>, Wiley, 1999",
        "isolators, their models, and the codes' procedures for isolated buildings",
    ),
    Reference(
        "chopra",
        "Chopra 2017",
        "A. K. Chopra, <i>Dynamics of Structures: Theory and Applications to Earthquake Engineering</i>, 5th "
        "edition, Pearson, 2017",
        "response spectra, modal combination and inelastic spectra, with the US codes and Eurocode 8",
    ),
    Reference(
        "wallace",
        "Wallace 1972",
        "C. E. Wallace, Radiation resistance of a rectangular panel, <i>Journal of the Acoustical Society of America</i> 51, 946–952, 1972",
        "",
    ),
    Reference(
        "maidanik",
        "Maidanik 1962",
        "G. Maidanik, Response of ribbed panels to reverberant acoustic fields, <i>Journal of the Acoustical Society of America</i> 34, 809–826, 1962",
        "",
    ),
    Reference(
        "crocker-price",
        "Crocker and Price 1969",
        "M. J. Crocker and A. J. Price, Sound transmission using statistical energy analysis, <i>Journal of Sound and Vibration</i> 9, 469–486, 1969",
        "",
    ),
    Reference(
        "lyon-maidanik",
        "Lyon and Maidanik 1962",
        "R. H. Lyon and G. Maidanik, Power flow between linearly coupled oscillators, <i>Journal of the Acoustical Society of America</i> 34, 623–639, 1962",
        "",
    ),
    Reference(
        "sabine",
        "Sabine 1922",
        "W. C. Sabine, <i>Collected Papers on Acoustics</i>, Harvard University Press, 1922 (Dover, 1964)",
        "his reverberation papers from 1900 on",
    ),
    Reference(
        "cremer-heckl",
        "Cremer, Heckl and Petersson 2005",
        "L. Cremer, M. Heckl and B. A. T. Petersson, <i>Structure-Borne Sound</i>, 3rd edition, Springer, 2005",
        "waves in structures, junctions, impedances and radiation: the reference for the theory",
    ),
    Reference(
        "lyon-dejong",
        "Lyon and DeJong 1995",
        "R. H. Lyon and R. G. DeJong, <i>Theory and Application of Statistical Energy Analysis</i>, 2nd edition, Butterworth-Heinemann, 1995",
        "SEA from the two oscillators up, with its coupling loss factors",
    ),
    Reference(
        "fahy-gardonio",
        "Fahy and Gardonio 2007",
        "F. Fahy and P. Gardonio, <i>Sound and Structural Vibration: Radiation, Transmission and Response</i>, 2nd edition, Academic Press, 2007",
        "radiation efficiency, transmission loss and fluid loading, the subject of this page",
    ),
    Reference(
        "craig-bampton",
        "Craig and Bampton 1968",
        "R. R. Craig and M. C. C. Bampton, Coupling of substructures for dynamic analyses, <i>AIAA Journal</i> 6, 1313–1319, 1968",
        "",
    ),
    Reference(
        "macneal",
        "MacNeal 1971",
        "R. H. MacNeal, A hybrid method of component mode synthesis, <i>Computers and Structures</i> 1, 581–601, 1971",
        "",
    ),
    Reference(
        "rubin",
        "Rubin 1975",
        "S. Rubin, Improved component-mode representation for structural dynamic analysis, <i>AIAA Journal</i> 13, 995–1006, 1975",
        "",
    ),
    Reference(
        "de-klerk",
        "de Klerk, Rixen and Voormeeren 2008",
        "D. de Klerk, D. J. Rixen and S. N. Voormeeren, General framework for dynamic substructuring: history, review and classification of techniques, <i>AIAA Journal</i> 46, 1169–1181, 2008",
        "the methods, modal and frequency-based, in one framework",
    ),
    Reference(
        "hurty",
        "Hurty 1965",
        "W. C. Hurty, Dynamic analysis of structural systems using component modes, <i>AIAA Journal</i> 3, 678–685, 1965",
        "",
    ),
    Reference(
        "guyan",
        "Guyan 1965",
        "R. J. Guyan, Reduction of stiffness and mass matrices, <i>AIAA Journal</i> 3, 380, 1965",
        "",
    ),
    Reference(
        "hou",
        "Hou 1969",
        "S.-N. Hou, Review of modal synthesis techniques and a new approach, <i>Shock and Vibration Bulletin</i> 40(4), 25–31, 1969",
        "",
    ),
    Reference(
        "goldman",
        "Goldman 1969",
        "R. L. Goldman, Vibration analysis by dynamic partitioning, <i>AIAA Journal</i> 7, 1152–1154, 1969",
        "",
    ),
    Reference(
        "craig-chang",
        "Craig and Chang 1977",
        "R. R. Craig and C.-J. Chang, <i>Substructure Coupling for Dynamic Analysis and Testing</i>, NASA CR-2781, 1977",
        "",
    ),
    Reference(
        "craig-kurdila",
        "Craig and Kurdila 2006",
        "R. R. Craig and A. J. Kurdila, <i>Fundamentals of Structural Dynamics</i>, 2nd edition, Wiley, 2006",
        "component mode synthesis, fixed and free interface, from the authors of the first",
    ),
    Reference(
        "turner",
        "Turner et al. 1956",
        "M. J. Turner, R. W. Clough, H. C. Martin and L. J. Topp, Stiffness and deflection analysis of complex "
        "structures, <i>Journal of the Aeronautical Sciences</i> 23, 805–823, 1956",
        "the direct stiffness method, and the finite element, for aircraft structures",
    ),
    Reference(
        "maxwell-frames",
        "Maxwell 1864",
        "J. C. Maxwell, On the calculation of the equilibrium and stiffness of frames, <i>Philosophical "
        "Magazine</i> 27, 294–299, 1864",
        "the count of bars a frame needs to be stiff",
    ),
    Reference(
        "calladine",
        "Calladine 1978",
        "C. R. Calladine, Buckminster Fuller's “tensegrity” structures and Clerk Maxwell's rules for the "
        "construction of stiff frames, <i>International Journal of Solids and Structures</i> 14, 161–172, 1978",
        "Maxwell's count made exact, with mechanisms and states of self-stress",
    ),
    Reference(
        "friedman-kosmatka",
        "Friedman and Kosmatka 1993",
        "Z. Friedman and J. B. Kosmatka, An improved two-node Timoshenko beam finite element, <i>Computers &amp; "
        "Structures</i> 47, 473–481, 1993",
        "",
    ),
    Reference(
        "cowper",
        "Cowper 1966",
        "G. R. Cowper, The shear coefficient in Timoshenko's beam theory, <i>Journal of Applied Mechanics</i> 33, "
        "335–340, 1966",
        "",
    ),
    Reference(
        "petyt",
        "Petyt 2010",
        "M. Petyt, <i>Introduction to Finite Element Vibration Analysis</i>, 2nd edition, Cambridge University "
        "Press, 2010",
        "element matrices for bars, beams and frames worked in full, consistent and lumped mass",
    ),
    Reference(
        "nyquist",
        "Nyquist 1928",
        "H. Nyquist, Certain topics in telegraph transmission theory, <i>Transactions of the AIEE</i> 47, 617–644, "
        "1928",
        "",
    ),
    Reference(
        "shannon",
        "Shannon 1949",
        "C. E. Shannon, Communication in the presence of noise, <i>Proceedings of the IRE</i> 37, 10–21, 1949",
        "",
    ),
    Reference(
        "butterworth",
        "Butterworth 1930",
        "S. Butterworth, On the theory of filter amplifiers, <i>Experimental Wireless and the Wireless "
        "Engineer</i> 7, 536–541, 1930",
        "",
    ),
    Reference(
        "cooley-tukey",
        "Cooley and Tukey 1965",
        "J. W. Cooley and J. W. Tukey, An algorithm for the machine calculation of complex Fourier series, "
        "<i>Mathematics of Computation</i> 19, 297–301, 1965",
        "",
    ),
    Reference(
        "harris-windows",
        "Harris 1978",
        "F. J. Harris, On the use of windows for harmonic analysis with the discrete Fourier transform, "
        "<i>Proceedings of the IEEE</i> 66, 51–83, 1978",
        "",
    ),
    Reference(
        "grossmann-morlet",
        "Grossmann and Morlet 1984",
        "A. Grossmann and J. Morlet, Decomposition of Hardy functions into square integrable wavelets of constant "
        "shape, <i>SIAM Journal on Mathematical Analysis</i> 15, 723–736, 1984",
        "",
    ),
    Reference(
        "iso-1683",
        "ISO 1683:2015",
        "ISO 1683:2015, <i>Acoustics: Preferred reference values for acoustical and vibratory levels</i>, "
        "International Organization for Standardization, 2015",
        "",
    ),
    Reference(
        "iec-61260",
        "IEC 61260-1:2014",
        "IEC 61260-1:2014, <i>Electroacoustics: Octave-band and fractional-octave-band filters, Part 1: "
        "Specifications</i>, International Electrotechnical Commission, 2014",
        "",
    ),
    Reference(
        "brandt",
        "Brandt 2011",
        "A. Brandt, <i>Noise and Vibration Analysis: Signal Analysis and Experimental Procedures</i>, Wiley, 2011",
        "the measurement side in full: sampling, windows, spectra and their errors, filters, transducers",
    ),
    Reference(
        "trethewey-cafeo",
        "Trethewey and Cafeo 1992",
        "M. W. Trethewey and J. A. Cafeo, Tutorial: signal processing aspects of structural impact testing, "
        "<i>International Journal of Analytical and Experimental Modal Analysis</i> 7(2), 129–149, 1992",
        "",
    ),
    Reference(
        "mcconnell-varoto",
        "McConnell and Varoto 2008",
        "K. G. McConnell and P. S. Varoto, <i>Vibration Testing: Theory and Practice</i>, 2nd edition, Wiley, 2008",
        "transducers, analyzers and exciters, and how the shaker and the structure interact",
    ),
    Reference(
        "vold-polyref",
        "Vold, Kundrat, Rocklin and Russell 1982",
        "H. Vold, J. Kundrat, G. T. Rocklin and R. Russell, A multi-input modal estimation algorithm for "
        "mini-computers, SAE Technical Paper 820194, 1982",
        "",
    ),
    Reference(
        "guillaume-lscf",
        "Guillaume et al. 2003",
        "P. Guillaume, P. Verboven, S. Vanlanduit, H. Van der Auweraer and B. Peeters, A poly-reference "
        "implementation of the least-squares complex frequency-domain estimator, <i>Proceedings of IMAC XXI</i>, 2003",
        "",
    ),
    Reference(
        "peeters-polymax",
        "Peeters et al. 2004",
        "B. Peeters, H. Van der Auweraer, P. Guillaume and J. Leuridan, The PolyMAX frequency-domain method: a new "
        "standard for modal parameter estimation?, <i>Shock and Vibration</i> 11, 395–409, 2004",
        "",
    ),
    Reference(
        "brincker-fdd",
        "Brincker, Zhang and Andersen 2001",
        "R. Brincker, L. Zhang and P. Andersen, Modal identification of output-only systems using frequency domain "
        "decomposition, <i>Smart Materials and Structures</i> 10, 441–445, 2001",
        "",
    ),
    Reference(
        "brincker-efdd",
        "Brincker, Ventura and Andersen 2001",
        "R. Brincker, C. E. Ventura and P. Andersen, Damping estimation by frequency domain decomposition, "
        "<i>Proceedings of IMAC XIX</i>, Kissimmee, 698–703, 2001",
        "",
    ),
    Reference(
        "james-next",
        "James, Carne and Lauffer 1995",
        "G. H. James, T. G. Carne and J. P. Lauffer, The natural excitation technique (NExT) for modal parameter "
        "extraction from operating structures, <i>Modal Analysis: the International Journal of Analytical and "
        "Experimental Modal Analysis</i> 10(4), 260–277, 1995",
        "",
    ),
    Reference(
        "peeters-deroeck",
        "Peeters and De Roeck 1999",
        "B. Peeters and G. De Roeck, Reference-based stochastic subspace identification for output-only modal "
        "analysis, <i>Mechanical Systems and Signal Processing</i> 13, 855–878, 1999",
        "",
    ),
    Reference(
        "jacobsen-harmonics",
        "Jacobsen, Andersen and Brincker 2006",
        "N.-J. Jacobsen, P. Andersen and R. Brincker, Using enhanced frequency domain decomposition as a robust "
        "technique to harmonic excitation in operational modal analysis, <i>Proceedings of ISMA2006</i>, Leuven, "
        "2006",
        "",
    ),
    Reference(
        "parloo-scaling",
        "Parloo et al. 2002",
        "E. Parloo, P. Verboven, P. Guillaume and M. Van Overmeire, Sensitivity-based operational mode shape "
        "normalisation, <i>Mechanical Systems and Signal Processing</i> 16, 757–767, 2002",
        "",
    ),
    Reference(
        "brincker-ventura",
        "Brincker and Ventura 2015",
        "R. Brincker and C. E. Ventura, <i>Introduction to Operational Modal Analysis</i>, Wiley, 2015",
        "OMA from random vibration up: FDD, SSI, record lengths, close modes and scaling",
    ),
    Reference(
        "fox-kapoor",
        "Fox and Kapoor 1968",
        "R. L. Fox and M. P. Kapoor, Rates of change of eigenvalues and eigenvectors, <i>AIAA Journal</i> 6(12), "
        "2426–2429, 1968",
        "",
    ),
    Reference(
        "nocedal-wright",
        "Nocedal and Wright 2006",
        "J. Nocedal and S. J. Wright, <i>Numerical Optimization</i>, 2nd edition, Springer, 2006",
        "",
    ),
    Reference(
        "tikhonov-arsenin",
        "Tikhonov and Arsenin 1977",
        "A. N. Tikhonov and V. Y. Arsenin, <i>Solutions of Ill-Posed Problems</i>, Winston, 1977",
        "",
    ),
    Reference(
        "friswell-mottershead",
        "Friswell and Mottershead 1995",
        "M. I. Friswell and J. E. Mottershead, <i>Finite Element Model Updating in Structural Dynamics</i>, Kluwer, "
        "1995",
        "correlation, sensitivities, regularization and the methods built on them",
    ),
    Reference(
        "fisher",
        "Fisher 1925",
        "R. A. Fisher, “Theory of statistical estimation”, <i>Proceedings of the Cambridge Philosophical Society</i> "
        "22(5), 700–725, 1925",
        "",
    ),
    Reference(
        "kammer",
        "Kammer 1991",
        "D. C. Kammer, “Sensor placement for on-orbit modal identification and correlation of large space "
        "structures”, <i>Journal of Guidance, Control, and Dynamics</i> 14(2), 251–259, 1991",
        "effective independence",
    ),
    Reference(
        "richardson-ods",
        "Richardson 1997",
        "M. H. Richardson, “Is it a mode shape, or an operating deflection shape?”, <i>Sound and Vibration</i>, "
        "March 1997 (30th anniversary issue)",
        "",
    ),
    Reference(
        "gelb-vandervelde",
        "Gelb and Vander Velde 1968",
        "A. Gelb and W. E. Vander Velde, <i>Multiple-Input Describing Functions and Nonlinear System Design</i>, "
        "McGraw-Hill, 1968",
        "",
    ),
    Reference(
        "juang",
        "Juang 1994",
        "J.-N. Juang, <i>Applied System Identification</i>, Prentice Hall, 1994",
        "Markov parameters",
    ),
    Reference(
        "hansen-picard",
        "Hansen 1990",
        "P. C. Hansen, “The discrete Picard condition for discrete ill-posed problems”, <i>BIT</i> 30(4), "
        "658–672, 1990",
        "",
    ),
    Reference(
        "hansen-lcurve",
        "Hansen 1992",
        "P. C. Hansen, “Analysis of discrete ill-posed problems by means of the L-curve”, <i>SIAM Review</i> "
        "34(4), 561–580, 1992",
        "",
    ),
    Reference(
        "worden-tomlinson",
        "Worden and Tomlinson 2001",
        "K. Worden and G. R. Tomlinson, <i>Nonlinearity in Structural Dynamics: Detection, Identification and "
        "Modelling</i>, Institute of Physics Publishing, 2001",
        "detecting nonlinearity in tests, and describing functions",
    ),
    Reference(
        "anderson",
        "Anderson 1958",
        "P. W. Anderson, “Absence of diffusion in certain random lattices”, <i>Physical Review</i> 109(5), "
        "1492–1505, 1958",
        "",
    ),
    Reference(
        "hodges",
        "Hodges 1982",
        "C. H. Hodges, “Confinement of vibration by structural irregularity”, <i>Journal of Sound and Vibration</i> "
        "82(3), 411–424, 1982",
        "",
    ),
    Reference(
        "whitehead",
        "Whitehead 1966",
        "D. S. Whitehead, “Effect of mistuning on the vibration of turbo-machine blades induced by wakes”, "
        "<i>Journal of Mechanical Engineering Science</i> 8(1), 15–21, 1966",
        "",
    ),
    Reference(
        "castanier-pierre",
        "Castanier and Pierre 2006",
        "M. P. Castanier and C. Pierre, “Modeling and analysis of mistuned bladed disk vibration: status and "
        "emerging directions”, <i>Journal of Propulsion and Power</i> 22(2), 384–396, 2006",
        "mistuned bladed discs: localization, the amplification's statistics, reduced models and intentional mistuning",
    ),
    Reference(
        "api-617",
        "API 617",
        "American Petroleum Institute, <i>API Standard 617: Axial and Centrifugal Compressors and "
        "Expander-compressors</i>, 9th edition, 2022",
        "",
    ),
    Reference(
        "walker",
        "Walker 2004",
        "D. N. Walker, <i>Torsional Vibration of Turbomachinery</i>, McGraw-Hill, 2004",
        "torsional analysis of turbomachinery trains: modelling, excitation, motor start-ups, testing and standards",
    ),
    Reference(
        "nestorides",
        "Nestorides 1958",
        "E. J. Nestorides (compiler), <i>A Handbook on Torsional Vibration</i>, Cambridge University Press, 1958",
        "the classic handbook on engine and machinery trains: inertias, stiffnesses, natural frequencies, "
        "stresses and dampers",
    ),
    Reference(
        "ueda",
        "Ueda 1979",
        "Y. Ueda, Randomly transitional phenomena in the system governed by Duffing's equation, <i>Journal of "
        "Statistical Physics</i> 20, 181–196, 1979",
        "",
    ),
    Reference(
        "poincare-celeste",
        "Poincaré 1892–1899",
        "H. Poincaré, <i>Les méthodes nouvelles de la mécanique céleste</i>, 3 vols., Gauthier-Villars, Paris, "
        "1892, 1893, 1899",
        "",
    ),
    Reference(
        "mathieu",
        "Mathieu 1868",
        "É. Mathieu, Mémoire sur le mouvement vibratoire d'une membrane de forme elliptique, <i>Journal de "
        "Mathématiques Pures et Appliquées</i> (2) 13, 137–203, 1868",
        "",
    ),
    Reference(
        "floquet",
        "Floquet 1883",
        "G. Floquet, Sur les équations différentielles linéaires à coefficients périodiques, <i>Annales "
        "scientifiques de l'École Normale Supérieure</i> (2) 12, 47–88, 1883",
        "",
    ),
    Reference(
        "ince",
        "Ince 1925",
        "E. L. Ince, Researches into the characteristic numbers of the Mathieu equation, <i>Proceedings of the "
        "Royal Society of Edinburgh</i> 46, 316–322, 1925–26",
        "",
    ),
    Reference(
        "van-der-pol-strutt",
        "van der Pol and Strutt 1928",
        "B. van der Pol and M. J. O. Strutt, On the stability of the solutions of Mathieu's equation, "
        "<i>Philosophical Magazine</i> (7) 5(27), 18–38, 1928",
        "",
    ),
    Reference(
        "nayfeh-mook",
        "Nayfeh and Mook 1979",
        "A. H. Nayfeh and D. T. Mook, <i>Nonlinear Oscillations</i>, Wiley, 1979 (Wiley Classics Library reprint 1995)",
        "the standard reference: harmonic balance, multiple scales, the Duffing and Mathieu equations in depth",
    ),
    Reference(
        "kovacic-brennan",
        "Kovacic and Brennan 2011",
        "I. Kovacic and M. J. Brennan (eds.), <i>The Duffing Equation: Nonlinear Oscillators and their "
        "Behaviour</i>, Wiley, 2011",
        "a whole book on this page's oscillator: its history, jumps, super- and subharmonics, chaos",
    ),
    Reference(
        "dalembert-1752",
        "d'Alembert 1752",
        "J. le R. d'Alembert, <i>Essai d'une nouvelle théorie de la résistance des fluides</i>, David l'aîné, "
        "Paris, 1752",
        "",
    ),
    Reference(
        "fritz",
        "Fritz 1972",
        "R. J. Fritz, The effect of liquids on the dynamic motions of immersed solids, <i>Journal of Engineering "
        "for Industry</i> (ASME) 94, 167–173, 1972",
        "",
    ),
    Reference(
        "lamb-1920",
        "Lamb 1920",
        "H. Lamb, On the vibrations of an elastic plate in contact with water, <i>Proceedings of the Royal Society "
        "of London A</i> 98, 205–216, 1920",
        "",
    ),
    Reference(
        "kwak",
        "Kwak 1991",
        "M. K. Kwak, Vibration of circular plates in contact with water, <i>Journal of Applied Mechanics</i> 58(2), "
        "480–483, 1991",
        "",
    ),
    Reference(
        "graham-rodriquez",
        "Graham and Rodriquez 1952",
        "E. W. Graham and A. M. Rodriquez, The characteristics of fuel motion which affect airplane dynamics, "
        "<i>Journal of Applied Mechanics</i> 19(3), 381–388, 1952",
        "",
    ),
    Reference(
        "abramson",
        "Abramson 1966",
        "H. N. Abramson (ed.), <i>The Dynamic Behavior of Liquids in Moving Containers</i>, NASA SP-106, 1966",
        "the classic survey of sloshing: tanks of every shape, their analogues and damping",
    ),
    Reference(
        "ibrahim",
        "Ibrahim 2005",
        "R. A. Ibrahim, <i>Liquid Sloshing Dynamics: Theory and Applications</i>, Cambridge University Press, 2005",
        "sloshing in depth: linear and nonlinear theory, analogues, liquid dampers",
    ),
    Reference(
        "van-der-pol-1926",
        "van der Pol 1926",
        "B. van der Pol, On “relaxation-oscillations”, <i>Philosophical Magazine</i> (7) 2(11), 978–992, 1926",
        "",
    ),
    Reference(
        "parkinson-smith",
        "Parkinson and Smith 1964",
        "G. V. Parkinson and J. D. Smith, The square prism as an aeroelastic non-linear oscillator, <i>Quarterly "
        "Journal of Mechanics and Applied Mathematics</i> 17(2), 225–239, 1964",
        "",
    ),
    Reference(
        "facchinetti",
        "Facchinetti, de Langre and Biolley 2004",
        "M. L. Facchinetti, E. de Langre and F. Biolley, Coupling of structure and wake oscillators in "
        "vortex-induced vibrations, <i>Journal of Fluids and Structures</i> 19(2), 123–140, 2004",
        "",
    ),
    Reference(
        "theodorsen",
        "Theodorsen 1935",
        "T. Theodorsen, General theory of aerodynamic instability and the mechanism of flutter, NACA Report 496, 1935",
        "",
    ),
    Reference(
        "blevins-fiv",
        "Blevins 1990",
        "R. D. Blevins, <i>Flow-Induced Vibration</i>, 2nd edition, Van Nostrand Reinhold, 1990 (Krieger reprint, "
        "corrected)",
        "galloping, vortex shedding, flutter and more, with the data to design against them",
    ),
    Reference(
        "dowell",
        "Dowell 2022",
        "E. H. Dowell (ed.), <i>A Modern Course in Aeroelasticity</i>, 6th edition, Springer, 2022",
        "flutter and divergence from the typical section to unsteady aerodynamics",
    ),
)
BY_KEY = {r.key: r for r in REFERENCES}
CITE = re.compile(r"<cite>(.*?)</cite>")
READING = re.compile(r"<reading>(.*?)</reading>", re.S)


def cite(key: str) -> str:
    """A citation in the text: a link to the reference's entry in the page's Further reading."""
    return f'<a href="#ref-{key}">{BY_KEY[key].cite}</a>'


def reading_html(keys: list[str]) -> str:
    """A page's Further reading section: `keys`' entries, in that order."""
    items = []
    for key in keys:
        r = BY_KEY[key]
        note = f": {r.note}" if r.note else ""
        items.append(f'<li><a name="ref-{key}"></a>{r.entry}{note}.</li>')
    return '<h3><a name="further-reading"></a>Further reading</h3><ul>' + "".join(items) + "</ul>"
