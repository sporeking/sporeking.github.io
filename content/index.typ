#import "../config.typ": template, tufted
#import "@preview/cmarker:0.1.7"
#show: template

= Jiong-Da Wang （王炯达）

#tufted.margin-note[
  // Statistician, Artist, and Professor Emeritus \
  Website: #link("https://sporeking.github.io")[sporeking.github.io] \
  Email: #link("mailto:wangjd@lamda.nju.edu.cn")[`wangjd@lamda.nju.edu.cn`]
]

Senior Undergraduate Student, LAMDA Group \
School of Intelligence Science and Technology \
State Key Laboratory of Novel Software Technology \
Nanjing University, Suzhou 215163, China \

Supervisor: Associate Professor #link("https://daiwz.net/")[Wang-Zhou Dai] \

== Research Interests

My current research interests mainly include neuro-symbolic learning. Specifically, I am interested in the following topics:

- Neuro-symbolic reasoning based on logic programming
- Combining sub-symbolic machine learning and symbolic machine learning (on noisy sub-symbolic data)

(If you're interested in any of these topics, feel free to reach out to me via email for further discussion!)

// == Artworks

// #tufted.margin-note[
//   #image("escaping-flatland.webp")
// ]

// == Books
// #{
//   let bib = load-bibliography(read("books.bib"))
//   for item in bib.values().rev() [
//     #let data = item.fields
//     - #strong(data.year): #emph(data.title)
//   ]
// }

// == Papers
// #{
//   let bib = load-bibliography(read("papers.bib"))
//   for item in bib.values().rev() [
//     #let data = item.fields
//     - #data.author, "#data.title," #emph(data.journal), #data.year. DOI: #link(data.url)[#data.doi]
//   ]
// }


// == Education
// - PhD in Political Science: Yale University (1968).
// - MS in Statistics: Stanford University.
// - BS in Statistics: Stanford University.
