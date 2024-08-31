Here’s the updated README with details on how you achieved your accuracies:

---

# EEG Data Analysis Using Deep Learning and Neural Networks

## Project Overview

This project focuses on analyzing EEG (Electroencephalogram) data using various Deep Learning and Neural Network models. Our team experimented with several architectures, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and transformers, among others. The goal was to identify the most effective model for classifying EEG data accurately.

## Team Members

- **Utkarsh Mittal**
- **Anvesha Dutta**
- **Avii Ahuja**
- **Praveena Ratnavel**

## Models Tested

We implemented and tested the following models, using a combination of data augmentation, hyperparameter tuning, and regularization techniques to enhance their performance:

1. **CNN (Best Model)** - `CNN_BEST (72.9%).ipynb`
   - Achieved the highest accuracy of 72.9% through extensive hyperparameter tuning and the application of regularization techniques such as dropout and batch normalization.
   
2. **EEGNet with Shallow and Deep Convolutional Layers** - `EEGNET+_Shallow_Conv+_Deep_Conv(70.2%).ipynb`
   - Achieved an accuracy of 70.2% by leveraging data augmentation and optimized convolutional layers.

3. **CNN with LSTM** - `CNN+LSTM_(67.94%).ipynb`
   - Achieved an accuracy of 67.94% through the combination of CNN for feature extraction and LSTM for sequence modeling, with careful tuning of hyperparameters.

4. **Final LSTM** - `Final_LSTM (63.7%).ipynb`
   - Achieved an accuracy of 63.7% using LSTM, enhanced by regularization techniques like L2 regularization.

5. **CNN with GRU** - `CNN+GRU(62.3%).ipynb`
   - Achieved an accuracy of 62.3% by integrating CNN and GRU, with fine-tuned learning rates and regularization.

6. **GRU** - `GRU (40.60%).ipynb`
   - Achieved an accuracy of 40.6%, which we improved as much as possible using data augmentation, though the model still underperformed.

7. **CNN with Transformers** - `CNN+Transformers(40.8%).ipynb`
   - Achieved an accuracy of 40.8% using a hybrid approach, though the complex model struggled with the given data, despite significant tuning efforts.

8. **Subject-wise CNN** - `CNN_Subject_wise.ipynb`
   - Accuracy varies based on individual subjects, with results enhanced by subject-specific tuning.

## How We Achieved Our Accuracies

To achieve the best possible accuracies, our approach included:

- **Data Augmentation**: Applied various augmentation techniques to increase the diversity of the training data, helping models generalize better to unseen data.
  
- **Hyperparameter Tuning**: Extensively experimented with different learning rates, batch sizes, number of layers, and neurons per layer to find the optimal settings for each model.

- **Regularization Techniques**: Implemented dropout, batch normalization, and L2 regularization to prevent overfitting and improve model performance.

## File Descriptions

- **Final_LSTM (63.7%).ipynb**: This notebook contains the implementation of an LSTM model which achieved 63.7% accuracy.
- **EEGNET+_Shallow_Conv+_Deep_Conv(70.2%).ipynb**: Implementation of EEGNet using shallow and deep convolutional layers, resulting in 70.2% accuracy.
- **GRU (40.60%).ipynb**: A GRU-based model, which unfortunately only achieved 40.6% accuracy.
- **CNN+Transformers(40.8%).ipynb**: A hybrid model combining CNN with transformers, resulting in 40.8% accuracy.
- **CNN+LSTM_(67.94%).ipynb**: A model combining CNN with LSTM layers, achieving 67.94% accuracy.
- **CNN+GRU(62.3%).ipynb**: This notebook features a combination of CNN and GRU, with 62.3% accuracy.
- **CNN_Subject_wise.ipynb**: A subject-wise CNN model with accuracy varying per individual.
- **CNN_BEST (72.9%).ipynb**: Our best-performing model, a CNN that achieved 72.9% accuracy.

## Conclusion

Through this project, we explored different neural network architectures to classify EEG data. The CNN model, with its carefully tuned hyperparameters and regularization techniques, proved to be the most effective, achieving the highest accuracy of 72.9%. Each model's implementation is documented in the respective Jupyter notebooks, where you can find the code and further details on the results.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/umittal10/EEG_Data_Analysis_NN.git
   ```
2. Open any of the Jupyter notebooks in your preferred environment:
   ```bash
   jupyter notebook CNN_BEST (72.9%).ipynb
   ```
3. Follow the instructions within each notebook to run the models and analyze the results.

## Future Work

- **Further Hyperparameter Tuning**: Additional tuning could yield even better results.
- **Advanced Data Augmentation**: Implement more sophisticated augmentation techniques.
- **Ensemble Methods**: Explore combining multiple models for potentially higher accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
