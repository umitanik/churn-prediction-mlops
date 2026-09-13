import numpy as np
from catboost import Pool


def top_drivers(pipeline, features, n=3):
    preprocessing = pipeline[:-1]
    model = pipeline.steps[-1][1]

    transformed = preprocessing.transform(features)
    names = preprocessing.get_feature_names_out()

    shap_values = model.get_feature_importance(Pool(transformed), type="ShapValues")
    contributions = shap_values[0, :-1]

    order = np.argsort(-np.abs(contributions))[:n]
    return [
        {"feature": str(names[i]), "contribution": round(float(contributions[i]), 4)}
        for i in order
    ]
