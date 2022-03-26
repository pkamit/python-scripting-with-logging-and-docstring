from utils.all_utils import prepare_data,save_plot
import pandas as pd
from utils.models import Perceptron


def main(data, m_name, p_name, eta, epochs):
    df_OR = pd.DataFrame(data)
    X, y = prepare_data(df_OR)
    model_or = Perceptron(eta=eta, epochs=epochs)
    model_or.fit(X, y)
    _ = model_or.total_loss()
    model_or.save(filename=m_name, model_dir="model")
    save_plot(df_OR, model_or, filename=p_name)


if __name__ ==  "__main__":
    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 1, 1, 1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=OR, m_name="or.model", p_name="or.png", eta=ETA, epochs=EPOCHS)


