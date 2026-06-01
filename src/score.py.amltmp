# # # # import json
# # # # import joblib
# # # # import numpy as np
# # # # import os

# # # # def init():
# # # #     global model

# # # #     model_path = os.path.join(
# # # #         os.getenv("AZUREML_MODEL_DIR"),
# # # #         "model.pkl"
# # # #     )

# # # #     model = joblib.load(model_path)

# # # # def run(raw_data):

# # # #     data = json.loads(raw_data)

# # # #     prediction = model.predict([data["data"]])

# # # #     return {
# # # #         "prediction": prediction.tolist()
# # # #     }

# # # import os
# # # import pandas as pd
# # # import joblib
# # # from azureml.core.model import Model

# # # def init():
# # #     global model

# # #     model_path = os.path.join(
# # #         os.getenv("AZUREML_MODEL_DIR"),
# # #         "model.pkl"
# # #     )

# # #     model = joblib.load(model_path)

# # # def run(mini_batch):

# # #     results = []

# # #     for file_path in mini_batch:

# # #         df = pd.read_csv(file_path)

# # #         preds = model.predict(df)

# # #         df["prediction"] = preds

# # #         results.append(df)

# # #     return pd.concat(results)

# # import os
# # import joblib
# # import pandas as pd

# # def init():
# #     global model

# #     model_path = os.path.join(
# #         os.getenv("AZUREML_MODEL_DIR"),
# #         "model.pkl"
# #     )

# #     model = joblib.load(model_path)

# # def run(mini_batch):

# #     results = []

# #     for file_path in mini_batch:

# #         df = pd.read_csv(file_path)

# #         preds = model.predict(df)

# #         df["prediction"] = preds

# #         results.append(df)

# #     return pd.concat(results)

# import os
# import joblib
# import pandas as pd

# def init():
#     global model

#     model_path = os.path.join(
#         os.getenv("AZUREML_MODEL_DIR"),
#         "model.pkl"
#     )

#     model = joblib.load(model_path)

# def run(mini_batch):

#     results = []

#     for file_path in mini_batch:

#         df = pd.read_csv(file_path)

#         preds = model.predict(df)

#         df["prediction"] = preds

#         results.append(df)

#     final_df = pd.concat(results)

#     output_dir = os.environ["AZUREML_BI_OUTPUT_PATH"]

#     output_file = os.path.join(output_dir, "predictions.csv")

#     final_df.to_csv(output_file, index=False)

#     return ["success"]

import os
import uuid
import joblib
import pandas as pd

FEATURES = [
    'Year',
    'Month',
    'DayofMonth',
    'DayOfWeek',
    'OriginAirportID',
    'DestAirportID',
    'DepDel15',
    'Cancelled',
    'DepDelay'
]

def init():
    global model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "model.pkl"
    )

    model = joblib.load(model_path)
    # global model, encoder

    # model_dir = os.getenv("AZUREML_MODEL_DIR")

    # model = joblib.load(
    #     os.path.join(model_dir, "model.pkl")
    # )

    # encoder = joblib.load(
    #     os.path.join(model_dir, "carrier_encoder.pkl")
    # )

def run(mini_batch):

    results = []

    for file_path in mini_batch:

        df = pd.read_csv(file_path)
        # df["Carrier"] = encoder.transform(df["Carrier"])
        preds = model.predict(df[FEATURES])

        df["prediction"] = preds

        results.append(df)

    final_df = pd.concat(results)

    output_dir = os.environ["AZUREML_BI_OUTPUT_PATH"]

    output_file = os.path.join(
        output_dir,
        f"predictions_{uuid.uuid4().hex}.csv"
    )

    final_df.to_csv(output_file, index=False)

    return [output_file]
