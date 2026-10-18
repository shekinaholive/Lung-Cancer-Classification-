# Reproducible inference notes

This folder-free helper is an optional, cleaner interface to the **existing hybrid model**, not a retrained or re-evaluated model.

## Run

Install compatible versions of **tensorflow**, **xgboost**, **numpy**, **pillow**, and **gradio** in a Python environment.

1. Train or locate **xgboost_hybrid_model.json** from \`Hybrid_Model.ipynb\`.
2. Place it next to \`app.py\`, or set the \`LUNG_XGBOOST_MODEL\` environment variable to its path.
3. Run \`python app.py\`.

The checkpoint is not shipped here; without it, predictions are unavailable. The feature extractor is reconstructed as **MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="avg")** to match the hybrid notebook. Its weights are downloaded by Keras if not cached.

## Important discrepancy corrected

The original training pipeline uses \`ImageDataGenerator(rescale=1./255)\`, but \`Gradio_Web_deployment.ipynb\` used MobileNetV2's \`preprocess_input\` (-1 to 1 range). **app.py follows the actual training transform (0 to 1)**.

For now, the original notebook remains unchanged to preserve experiment history. Do not interpret its previous Gradio outputs as reproducing the recorded 98% test performance.

The training folder names are **lung_aca**, **lung_n** and **lung_scc**, so the app reports probabilities in that order. Verify \`train_val_generator.class_indices\` against the original run if using artifacts from another training session.

**Validation status:** Syntax and static contract checks can be run without model assets. Numerical equivalence and runtime inference cannot be confirmed without the exact trained checkpoint and source images. This is a research demonstration, not a diagnostic application.
