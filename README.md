# [tensorflow-haskell-deptyped][]

Hi, this repo is dedicated to experiment adding Dependent Types to [TensorFlow Haskell][].

Beware! This code is alpha stage, the API is not stable and will not be until merged to
the official project ([TensorFlow Haskell][]), hopefully.

### Making sure everything is alright
``` sh
stack setup
```

### Build the project.
``` sh
stack build
```

### Executing example
``` sh
stack exec -- tensorflow-haskell-deptyped
```

The example's code can be found in `executable/Main.hs`

[tensorflow-haskell-deptyped]: https://github.com/helq/tensorflow-haskell-deptyped
[TensorFlow Haskell]: https://github.com/tensorflow/haskell
