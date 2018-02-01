from __init__ import *

class double_norm_gen(rv_continuous):
    """ Double Gauss Funciton
    """
    def _pdf(self, x, mu1, mu2, sigma1, sigma2, f):
        return f * norm.pdf(x, loc = mu1, scale = sigma1) + (1-f) * norm.pdf(x, loc = mu2, scale = sigma2)

    def _cdf(self, x, mu1, mu2, sigma1, sigma2, f):
        return f * norm.cdf(x, loc = mu1, scale = sigma1) + (1-f) * norm.cdf(x, loc = mu2, scale = sigma2)
double_norm = double_norm_gen(name='double_norm')

class dcb_gen(rv_continuous):
    """ Double Crystall Ball Function
    """
    def _pdf_unormalised(self, x, mu, sigma, alpha_high, alpha_low, n_high, n_low):
        try :
          x_norm = ( (x - mu) / sigma )
          if (x_norm < -alpha_low ) :
            A = math.exp(-0.5*alpha_low*alpha_low)
            B = n_low/math.fabs(alpha_low) - math.fabs(alpha_low)
            output = A * math.pow( math.fabs(alpha_low)/n_low * (B - x_norm), -n_low )
          elif (x_norm > alpha_high) :
            A = math.exp(-0.5*alpha_high*alpha_high)
            B = n_high/alpha_high - alpha_high
            output = A * math.pow( math.fabs(alpha_high)/n_high * (B + x_norm), -n_high )
          else :
            output = math.exp(-0.5*x_norm*x_norm)
          return output
        except ValueError :
          return 0
    def _pdf(self, x, *args):
        integral = self._integrate_unnormalised_pdf( parameters=(mu, sigma, alpha_low, alpha_high, n_low, n_high) )
        if integral != 0 : return normalisation * self._unnormalised_pdf( x, **args ) / integral
dcb = dcb_gen(name='dcb')

class expo_gen(rv_continuous):
    """
    Exponetial pdf for the background fit 
    """
    def _pdf(self, x, tau):
        return tau * math.exp(-tau * x)
expo = expo_gen(name="expo")
        

