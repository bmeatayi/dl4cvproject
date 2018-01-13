class MDN2D(nn.Module):
    def __init__(self, in_height=None, in_width=None, hidden_size=256, num_mixtures=16):
        super(MDN2D, self).__init__()
        
        # an imgage has three channels
        # for videos, we have:
        # many images (clips) with 1 channel. This ends up with a saliency map for each clip, which is what we want!
        # So, REPLACE 3 BY 1 IN THE COMPLETE LETWROK
        
        self.conv2fc = nn.Conv2d(3, hidden_size, (in_height, in_width))
        self.tanh = nn.Tanh()
        
        self.mu_out = nn.Linear(hidden_size, num_mixtures)
        self.sigma_out = nn.Linear(hidden_size, num_mixtures)
        self.corr_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Tanh()
            )
        self.pi_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Softmax()
            )
        
    def forward(self, x):
        out = self.conv2fc(x)
        out = self.tanh(out)
        out = out.view(-1, self.num_flat_features(out))
        out_pi = self.pi_out(out)
        out_mu = self.mu_out(out)
        out_sigma = torch.exp(self.sigma_out(out))
        out_corr = self.corr_out(out)
        
        return (out_pi, out_mu, out_sigma, out_corr)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
