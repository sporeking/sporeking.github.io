#import "../index.typ": template, tufted
#show: template
#import "@preview/citegeist:0.2.0": load-bibliography

= Publications

== Conference Papers
#{
  let bib = load-bibliography(read("papers.bib"))
  for item in bib.values().rev() [
    #let data = item.fields
    - #data.author, "#data.title," #emph(data.booktitle), #data.year. DOI: #link(data.url)[#data.doi]
  ]
}

