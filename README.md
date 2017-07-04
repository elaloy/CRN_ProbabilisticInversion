# Probabilistic inversion of CRN depth profile data

This Python 2.7 package contains the (1) Markov chain Monte Carlo (MCMC) and (2) lattice-based Monte Carlo codes 
used for probabilistic inversion of cosmogenic radionuclide (CRN) depth profile data in [Laloy et al. (2017)](https://doi.org/10.5194/esurf-5-331-2017).

More information can be found in the run_mcmc.py and run_mc.py scripts of the 'MCMC_Inversion' and 'LatticeBasedMC_Inversion' folders, respectively.

## Performing the inversion

To perform the inversion, and postprocess its results (including figure creation), one can conveniently run the different sections (e.g., using Spyder) 
in run_mcmc.py for the MCMC inversion and run_mc.py for the lattice-based MC inversion.

## Citation

Please make sure to cite our paper, if you use any of the contained code in your own projects or publication: 

Laloy, E., Beertem, K., Vanacker, V., Christl, M., Rogiers, B., Wauters, L., 
    Bayesian inversion of a CRN depth profile to infer Quaternary erosion of the 
    northwestern Campine Plateau (NE Belgium), Earth Surf. Dynam., 5, 331–345, 
    2017, https://doi.org/10.5194/esurf-5-331-2017.

## License

Licence: the MCMC inversion code is under GPL license while the lattice-based MC code is under MIT license. See the corresponding folders for details.

## Contact

Eric Laloy (elaloy@sckcen.be) 