from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Hyperparameters
input_dim = 39  # input rates
output_dim = 200  # output rates
hidden_dim = 256  # hidden layer size

##########################################################################################
def price_rate(price_row):
    rate_list = []
    i = 0

    while i < len(price_row)-1:
        rate = price_row[i+1] - price_row[i]
        rate_list.append(rate)

        i+=1
    
    if len(rate_list) != len(price_row)-1:
        print(f'Error price rate len {len(rate_list)}')

    return rate_list

def rate_to_price(rate_list, first_price, tri):
    price_list = []
    if tri == 1:
        price_list.append(first_price)

    current_price = first_price
    
    for rate in rate_list:
        next_price = current_price + rate
        price_list.append(next_price)
        current_price = next_price

    return price_list

def data_plot(X_input_array, full_prices, X_output):
    full_pred_array = X_input_array + X_output
    full_prices_smoothed = smooth_sequence(full_prices)
    full_pred_array = smooth_sequence(full_pred_array)
    # Plotting
    print("full_prices:", np.array(full_prices).shape)
    print("full_pred_array:", np.array(full_pred_array).shape)
    plt.figure(figsize=(12, 6))

    # Plot real data (concatenation of input and expected output)
    plt.plot(full_prices, label='Real Data', alpha=0.5, linewidth=1)
    plt.plot(full_prices_smoothed, label='Real Data smoothed', linewidth=1)

    # Plot predicted data (concatenation of input and predicted output)
    plt.plot(full_pred_array, label='Predicted Data', linewidth=1)

    plt.xlabel('Time Point')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def smooth_sequence(X):
    return np.convolve(X, np.ones(5)/5, mode='valid')

csv_path = 'ml/GAN/historical_stock_data_vol.csv'
df = pd.read_csv(csv_path)
df_price = df.iloc[::2, :].reset_index(drop=True)
df_vol = df.iloc[1::2, :].reset_index(drop=True)
df_price = df_price * 100
df_vol = df_vol * 100
# df_price = df_price.apply(smooth_df, axis=1)
# df_vol = df_vol.apply(smooth_df, axis=1)

X_price_input = []
X_vol_input = []
first_prices = []  # List to hold the first actual price for each sequence
y_expected = [] 
full_prices = []

for index, row in df_price.iterrows():
    first_price = row.iloc[0]  # Store the first actual price 
    first_prices.append(first_price)
    full_prices = row
    
    price_rate_arr = price_rate(row)
    X_day_price = price_rate_arr[:39]
    y_day_expected = price_rate_arr[39:]

    X_price_input.append(X_day_price)
    y_expected.append(y_day_expected)

for index, row in df_vol.iterrows():
    X_day_vol = price_rate(row.iloc[:40]) 
    X_vol_input.append(X_day_vol)

print(f"X price shape : {np.array(X_price_input).shape}")
print(f"X price shape : {np.array(X_vol_input).shape}")
X_input = np.array(X_price_input)
# X_input = np.hstack((np.array(X_price_input), np.array(X_vol_input)))
print(f"X input shape : {X_input.shape}")

####################################################################################################

# Create the Generator
generator = Sequential([
    Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(hidden_dim, activation='relu'),
    Dropout(0.2),
    Dense(output_dim, activation='tanh')
])

# Create the Discriminator
discriminator = Sequential([
    Dense(hidden_dim, activation='relu', input_shape=(output_dim,)),
    Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combine the generator and the discriminator
discriminator.trainable = False
gan_input = Input(shape=(input_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Generate Training Data (Replace this with your real stock rates)
# X_train would be your array of first 39 rates, y_train would be the following 200 rates
X_train = X_input
y_train = np.array(y_expected)

# Training Loop
epochs = 1500
batch_size = 32

def train_nn():
    for epoch in range(epochs):
        
        # Train Discriminator
        noise = np.random.rand(batch_size, input_dim)
        generated_data = generator.predict(noise)
        real_data = y_train[np.random.randint(0, y_train.shape[0], batch_size)]
        
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_data, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_data, labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.rand(batch_size, input_dim)
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)
        
        print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")


    discriminator.save(f"ml/GAN/discriminator_model_{epochs}.h5")
    generator.save(f"ml/GAN/generator_model_{epochs}.h5")


def test_prediction_rate(models, csv_path):
    df = pd.read_csv(csv_path)
    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)
    df_price = df_price * 100
    df_vol = df_vol * 100
    # df_price = df_price.apply(smooth_df, axis=1)
    # df_vol = df_vol.apply(smooth_df, axis=1)

    X_price_input = []
    X_vol_input = []
    first_prices = []  # List to hold the first actual price for each sequence
    y_expected = [] 
    full_prices = []

    for index, row in df_price.iterrows():
        first_price = row.iloc[0]  # Store the first actual price 
        first_prices.append(first_price)
        full_prices = row
        
        price_rate_arr = price_rate(row)
        X_day_price = price_rate_arr[:39]
        y_day_expected = price_rate_arr[39:]

        X_price_input.append(X_day_price)
        y_expected.append(y_day_expected)
    
    for index, row in df_vol.iterrows():
        X_day_vol = price_rate(row.iloc[:40]) 
        X_vol_input.append(X_day_vol)
    
    print(f"X price shape : {np.array(X_price_input).shape}")
    print(f"X price shape : {np.array(X_vol_input).shape}")
    X_input = np.array(X_price_input)
    # X_input = np.hstack((np.array(X_price_input), np.array(X_vol_input)))
    print(f"X input shape : {X_input.shape}")
    
    y_predicted_rates = models.predict(X_input) 
    print(f"y_expected size : {np.array(y_expected).shape}")
    print(f"y_predicted_rates size : {np.array(y_predicted_rates).shape}")

    X_price_converted = []

    # print("Test Mean Squared Error:", mean_squared_error(y_expected_sliced, y_predicted_price))
    X_price_converted = rate_to_price(np.array(X_price_input[0]), first_price, 1)
    print(X_price_converted)
    print("X_price_converted:",np.array(X_price_converted).shape)
    
    first = X_price_converted[-1]

    y_predicted_price = rate_to_price(np.array(y_predicted_rates[0]), first, 2)
    print("y_predicted_price:",np.array(y_predicted_price).shape)
    # Assuming data_plot plots actual vs predicted sequences
    data_plot(X_price_converted, full_prices, y_predicted_price)


if __name__ == '__main__':
    # train_nn()
    loaded_discriminator = load_model("ml/GAN/discriminator_model_1500.h5")
    loaded_generator = load_model("ml/GAN/generator_model_1500.h5")
    csv_path = 'ml/test_data4.csv'

    predicted_output = test_prediction_rate(loaded_generator, csv_path)