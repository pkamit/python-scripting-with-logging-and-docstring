from utils.all_utils import prepare_data,save_plot
import pandas as pd
from utils.models import Perceptron


def main(data, m_name, p_name, eta, epochs):
    df = pd.DataFrame(data)
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    model.save(filename=m_name, model_dir="model")
    save_plot(df, model, filename=p_name)


if __name__ ==  "__main__":
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=AND, m_name="and.model", p_name="and.png", eta=ETA, epochs=EPOCHS)


