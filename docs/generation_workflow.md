# Generation Workflow

This stage turns retrieved pages into answer candidates using multiple visual reasoning strategies.

## Current Strategies

- `single_page`
  Answer from the top-1 retrieved page only.

- `stitched`
  Stitch top retrieved pages into one image, then answer from the combined image.

- `per_page_weighted`
  Generate one answer per page, estimate confidence, then combine generation confidence with retrieval score.

- `multi_image_joint`
  Feed multiple retrieved pages together for joint reasoning.

## Recommended Command

```bash
python scripts/generation/minimal_multimodal_generator.py --limit 3
```

## Main Output

- `outputs/generator_cases/multistrategy_generator_results.json`

Each query result includes all strategy outputs plus a final selected answer.
