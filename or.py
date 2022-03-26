import os.path

from utils.all_utils import prepare_data,save_plot
import pandas as pd
from utils.models import Perceptron
import logging

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s] %(message)s',
    filemode='a'
)


def main(data, m_name, p_name, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is the raw dataset: {df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    model.save(filename=m_name, model_dir="model")
    save_plot(df, model, filename=p_name)


if __name__ ==  "__main__":
    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 1, 1, 1]
    }
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info(">>>>>>>>>> starting traning >>>>>>>>")
        main(data=OR, m_name="or.model", p_name="or.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<< done training >>>>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e

