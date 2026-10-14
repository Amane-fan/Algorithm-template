// Four print-ready cover candidates. Compile with --root .
#import "styles/book.typ": book-cover
#let config = toml("templates.toml")
#set document(title: "Amane - Cover studies", author: config.book.author)
#for design in ("a", "b", "c", "d") {
  book-cover(config.book, config.layout, design: design)
}
