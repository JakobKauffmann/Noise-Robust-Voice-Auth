#!/usr/bin/env Rscript
# Visualization Module:
# Reads the training history CSV and plots training loss and accuracy using ggplot2.

library(ggplot2)
library(readr)

# Read the training history CSV.
history <- read_csv("training_history.csv")

# Plot Loss over Epochs.
loss_plot <- ggplot(history, aes(x = epoch, y = loss)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "blue", size = 2) +
  ggtitle("Training Loss over Epochs") +
  xlab("Epoch") +
  ylab("Loss") +
  theme_minimal()

# Plot Accuracy over Epochs.
accuracy_plot <- ggplot(history, aes(x = epoch, y = accuracy)) +
  geom_line(color = "green", size = 1) +
  geom_point(color = "green", size = 2) +
  ggtitle("Training Accuracy over Epochs") +
  xlab("Epoch") +
  ylab("Accuracy") +
  theme_minimal()

# Save the plots.
ggsave("loss_plot.png", plot = loss_plot, width = 6, height = 4)
ggsave("accuracy_plot.png", plot = accuracy_plot, width = 6, height = 4)

print("Plots saved as loss_plot.png and accuracy_plot.png")