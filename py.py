import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1: Generate a NumPy array for PM2.5 levels
np.random.seed(0) 
minutes = 1440
base_pm25 = np.random.uniform(0, 200, 1440)
noise = np.random.normal(0, 10, minutes)  
pollution_data = base_pm25 + noise

# 2: Apply a low-pass filter to reduce noise
def low_pass_filter(data, cutoff_freq=0.1, fs=1.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

filtered_data = low_pass_filter(pollution_data)

# 3: Compute and display average PM2.5 levels for each hour
def compute_hourly_averages(data):
    hourly_averages = np.mean(data.reshape(-1, 60), axis=1)
    return hourly_averages

hourly_averages = compute_hourly_averages(pollution_data)

for hour, avg in enumerate(hourly_averages):
    print(f"Hour {hour}: Average PM2.5 = {avg:.2f}")

# 4: Plot the original and filtered data
plt.figure(figsize=(14, 7))
plt.plot(pollution_data, label='Original Noisy Data', color='blue', linestyle='--')
plt.plot(filtered_data, label='Filtered Data', color='green')
threshold = 150
hazardous_hours = hourly_averages > threshold
for hour in range(24):
    if hazardous_hours[hour]:
        plt.axvspan(hour * 60, (hour + 1) * 60, color='red', alpha=0.3)

plt.title('PM2.5 Levels Over 24 Hours')
plt.xlabel('Minutes')
plt.ylabel('PM2.5 Level')
plt.legend()
plt.grid()
plt.show()

# 5: Detect and plot time intervals where PM2.5 > 200 for more than 10 consecutive minutes
exceedances = pollution_data > 200
consecutive_minutes = []
current_streak = 0

for minute in range(minutes):
    if exceedances[minute]:
        current_streak += 1
    else:
        if current_streak > 10:
            consecutive_minutes.append((minute - current_streak, current_streak))
        current_streak = 0

plt.figure(figsize=(14, 7))
plt.plot(pollution_data, label='PM2.5 Levels', color='blue')
plt.axhline(y=200, color='red', linestyle='--', label='200 µg/m³ Threshold')

for start, length in consecutive_minutes:
    plt.axvspan(start, start + length, color='orange', alpha=0.5, label='Exceeding 200 µg/m³' if start == consecutive_minutes[0][0] else "")

plt.title('PM2.5 Levels with Peaks Above 200 µg/m³')
plt.xlabel('Minutes')
plt.ylabel('PM2.5 Level')
plt.legend()
plt.grid()
plt.show()
