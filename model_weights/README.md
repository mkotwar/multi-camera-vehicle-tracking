# Model Weights

Expected Step 3 vehicle detector filename:

- `best_old.pt`

Verified source used for the local Step 3 rebuild:

- `F:\vinfo\Final_vedio_Ai_system\object\vehical_detection\best_old.pt`

Runtime configuration in this repo expects:

- `model_weights/best_old.pt`

How another user should set this up on a different machine:

1. Place the detector weight file under `model_weights/`
2. Keep the filename expected by `config.yaml`, or update `config.yaml`
3. Run the app from the repository root

Important:

- Git does not include `.pt` model files because `*.pt` is ignored.
- If `best_old.pt` is missing, startup will raise a clear error showing the resolved absolute expected path.
