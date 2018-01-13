class MDN2D(nn.Module):

    def __init__(self, in_height=None, in_width=None, hidden_size=256, num_mixture=16):
        super(MDN2D, self).__init__()
        
        self.conv2fn = nn.Conv2d(1, hidden_size, (in_height, in_width))
        self.tanh = nn.Tanh()
        
        self.mu_out = nn.Linear(hidden_size, num_mixtures)
        self.sigma_out = nn.Linear(hidden_size, num_mixtures)
        self.cor_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            torch.nn.Tanh()
        self.pi_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Softmax()
            )
        
    def forward(self, x):
        out = self.fc_in(x)
        out = self.tanh(out)
        out_pi = self.pi_out(out)
        out_mu = self.mu_out(out)
        out_sigma = torch.exp(self.sigma_out(out))
        out_cor = self.cor_out(out)
        
        return (out_pi, out_mu, out_sigma, out_cor)
        
