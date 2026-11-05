# Indigenous Knowledge and Geospatial Bias in the Era of Large Language Models

Companion website and materials for the **ACM SIGSPATIAL 2026 Tutorial** (November 3-6, 2026, Riverside, CA, USA).

- **Presenters**: Nicole R. Schneider (University of Maryland), Nandini Ramachandran (University of Maryland), Kent O'Sullivan (University of Sydney), Hanan Samet (University of Maryland)
- **Tutorial Paper**: [https://doi.org/10.1145/3841645.3843439](https://doi.org/10.1145/3841645.3843439)

## Repository Structure

- `index.html`: Static tutorial webpage (zero build step, ready for GitHub Pages)
- `slides/sigspatial26-tutorial-slides.pdf`: Tutorial presentation slides (PDF)
- `slides/sigspatial26-tutorial-slides.pptx`: Tutorial presentation slides (PowerPoint)
- `data/dataset.csv`: Original Australian dual-named place dataset from Pyda, Schneider, and Samet (GeoAI '25, [DOI: 10.1145/3764912.3770811](https://doi.org/10.1145/3764912.3770811))
- `data/dual_named_places_paired.csv`: Paired table of the 13 Indigenous and Anglicized Australian city toponyms with 2021 census populations
- `code/experiment.py`: Embedding analysis, t-SNE projection, and accuracy correlation script from [nicoleschneider/LLM-Embeddings](https://github.com/nicoleschneider/LLM-Embeddings)

## Publishing on GitHub Pages

1. Create a public GitHub repository (for example, `sigspatial26-tutorial`) and push or upload the contents of this directory to the `main` branch.
2. Open **Settings -> Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**, select `main` and `/ (root)`, and click **Save**.
4. Your site will be publicly accessible at `https://<your-github-username>.github.io/sigspatial26-tutorial/`.
