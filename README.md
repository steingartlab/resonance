# resonance

This is the core codebase needed for analyzing resonance experiments. The software for conducting experiments is not publically available, more out of lack of refactoring it in an presentable manner than anything else.

It has the functionality needed to load, parse and perform standard resonance analysis on data from both potentiostats and oscilloscopes.

It also includes a hodgepodge of classes and functions to construct and assemble figures.

It is designed to be called from the omnipotent [pithy](https://github.com/dansteingart/pithy) and use the file management system [drops](https://github.com/dansteingart/drops), both written by proftron Dan himself. 

```
├── LICENSE
├── README.md
├── resonance
│   ├── backend.py
│   ├── dsp.py
│   ├── figures.py
│   ├── limit_memory.py
│   ├── potentiostats.py
│   ├── pressure.py
│   ├── resonance.py
│   ├── resostat.py
│   └── utils.py
└── tests
    ├── test_backend.py
    ├── test_dsp.py
    ├── test_limit_memory.py
    ├── test_potentiostats.py
    ├── test_pressure.py
    ├── test_resonance.py
    └── test_resostat.py

```

TODO: unittests that aren't figures