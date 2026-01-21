# Conference Reader - Issues & Improvements

## Project Context
Currently working with test images to validate the image → summary text flow.

---

## Critical Issues

### Test Image Processing
- [*] ~~Issue: Something is wrong with the prompting such that the first document processed always return a summary text = to the example text. We need to fix that so it only generates the summary text based on the extracted text NOT the example~~
- [ ] Issue: Image /data/neurips/poster_test/IMG_1223.JPEG is a picture that was taken to isolate the QR codes for the poster and does not represent a complete poster. We need to determine if the model can detect and exclude these images or if we need to manually filter them before processing the directory
- [ ] Issue: [Description needed]
- [ ] Issue: [Description needed]
---

## Quality Improvements

- [ ] Rather than filtering in summarize_single or summarize_batch for doc failures or docs with no extracted text, we should handle this at the doc extraction phase. The list of documents returned for further processing should only include valid documents. Invalid documents should be written to a file output/extraction_failures.json which is a json document with a list of failed documents including all their metadata. We should use the output package to handle writing this file. The main program, when it reaches the end, should output to console the number of failed documents at the extraction phase. 
- [ ] Handle HEIC images which will be better quality as the source images
- [ ] [To be added]
- [ ] [To be added]

---

## Performance Concerns

- [ ] [To be added]

---

## Feature Requests / Enhancements

- [ ] [To be added]

---

## Resolved Issues

### [Date] - Issue Title
- **Problem**: Description
- **Solution**: What was done
- **Files affected**: List of files

---

## Notes

Add any additional context, observations, or technical decisions here.
Image /data/neurips/poster_test/IMG_1212.JPEG can not be parsed for an accurate title because there is a flag covering the first letters of the title. This is not an error of the program, this is a source data error and should be left alone