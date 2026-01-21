from docling.document_converter import DocumentConverter


source = "/data/neurips/poster_test/bad-test.jpg"

converter = DocumentConverter()
result = converter.convert(source)


print(result.document.export_to_markdown())
