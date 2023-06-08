## Don't change the Model Support matrix in README.md manually!

1. Create a feature branch
2. Add new entry to `automation/models.csv` file.
3. Run `automation/generate-model-matrix.js` in this `automation` directory.
4. Model support matrix in `../README.md` will update automatically.

```bash
node automation/generate-model-matrix.js
```

5. Commit changes to your branch. Create PR documenting that you're working on the model.
6. Implement the model.
7. Submit another PR to update the table once your model is completed.