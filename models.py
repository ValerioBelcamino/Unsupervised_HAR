import torch.nn as nn
import torch

class LSTMMultiClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob=0.5):
        super(LSTMMultiClass, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, dropout=dropout_prob, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, output_dim)
        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sm(out)
        return out


class LSTMBinary(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob=0.5):
        super(LSTMBinary, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, dropout=dropout_prob, batch_first=True)
        # self.fc1 = nn.Linear(hidden_dim, 128)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(128, output_dim)
        # self.sm = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # out = self.dropout(lstm_out[:, -1, :])
        out = self.dropout(lstm_out[:, -1, :])
        # out = self.relu(out)
        out = self.fc(out)
        out = self.sig(out)
        return out


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_dim=128, n_layers=2, n_heads=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = n_features, nhead = n_heads),
            num_layers = n_layers
            )
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_features, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    

class CNN_1D(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 13),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 9),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_prob),
            nn.Flatten(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	nn.Dropout(dropout_prob),
        	nn.Linear(23616, 128),
        	nn.ReLU(),
        	nn.Dropout(dropout_prob),
        	nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        	)

    def forward(self, x):
        x = self.features(x.permute(0, 2, 1))
        # print(x.shape)
        
        # x = x.view(x.size(0), 1792)
        out = self.classifier(x)

        return out


class CNN_1D_multihead(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super().__init__()

        '''Head 1'''
        # Extract features, 1D conv layers
        self.head1 = nn.Sequential(
            nn.Conv1d(input_size, 64, 17), #17
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 13),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten()
        )

        '''Head 2'''
        self.head2 = nn.Sequential(
            nn.Conv1d(input_size, 64, 11),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 9),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten()
        )

        '''Head 3'''
        self.head3 = nn.Sequential(
            nn.Conv1d(input_size, 64, 7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(4),            
            nn.Conv1d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten()
        )

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	nn.Dropout(dropout_prob),
        	nn.Linear(1024, 128), #143552 #35520 #8640 #1088
        	nn.ReLU(),
        	nn.Dropout(dropout_prob),
            # nn.Linear(1024, 128),
        	# nn.ReLU(),
        	# nn.Dropout(dropout_prob),
        	nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.head1(x.permute(0, 2, 1))
        x2 = self.head2(x.permute(0, 2, 1))
        x3 = self.head3(x.permute(0, 2, 1))

        x = torch.cat((x1, x2, x3), 1)
        # print(x.shape)
        
        # x = x.view(x.size(0), 1792)
        out = self.classifier(x)

        return out