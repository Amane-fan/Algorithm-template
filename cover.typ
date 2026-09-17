// Standalone cover: uses the same design and metadata as the complete book.
#import "styles/book.typ": book-cover
#let config = toml("templates.toml")
#set document(title: config.book.title, author: config.book.author)
#book-cover(config.book, config.layout, cover: if config.cover.image == "" {
  none
} else {
  image(config.cover.image, width: 178mm, height: 119mm, fit: "contain")
})
