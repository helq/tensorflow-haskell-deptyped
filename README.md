# tensorflow-haskell-deptyped #

This repo is dedicated to experiment adding Dependent Types to [TensorFlow Haskell][].

Beware! The API is not yet stable. This code is in alpha stage.

This repo may be merged into [TensorFlow Haskell][] in the future, in the meantime is just
playground to test dependent types and tensorflow.

[TensorFlow Haskell]: https://github.com/tensorflow/haskell

## How to run ##

**Making sure everything is alright**:

``` sh
stack setup
```

**Build the project**:

``` sh
stack build
```

**Running the examples**:

``` sh
stack exec -- tensorflow-haskell-deptyped
stack exec -- tf-example
```

## Examples ##

Some simple examples can be found in [Main.hs](executable/Main.hs).

A more complete example using MNIST can be found in [tensorflow-minst-deptyped](tensorflow-mnist-deptyped/app/Main.hs).

Imitating the presentation shown in <https://github.com/tensorflow/haskell> below a
minimal example of using TensorFlow in Haskell (with Dependent Types)
\[[full code](executable/tf-haskell-example.hs)\]:

```haskell
{-# LANGUAGE DataKinds, TypeApplications, ScopedTypeVariables #-}

import Control.Monad (replicateM_)
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Minimize as TF

import qualified TensorFlow.DepTyped as TFD
import           Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as VS (replicateM, map)

import           GHC.TypeLits (KnownNat)

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- VS.replicateM @100 randomIO
    let yData = VS.map (\x->x*3 + 8) xData
    -- Fit linear regression model.
    (w, b) <- fit xData yData
    assertBool "w == 3" (abs (3 - w) < 0.001)
    assertBool "b == 8" (abs (8 - b) < 0.001)

fit :: forall n. KnownNat n => Vector n Float -> Vector n Float -> IO (Float, Float)
fit xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TFD.constant @'[n] xData
        y = TFD.constant @'[n] yData
    -- Create scalar variables for slope and intercept.
    w <- TFD.initializedVariable @'[1] 0
    b <- TFD.initializedVariable @'[1] 0
    -- Define the loss function.
    let yHat = (x `TFD.mul` TFD.readValue w) `TFD.add` TFD.readValue b
        loss = TFD.square (yHat `TFD.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TFD.minimizeWith (TF.gradientDescent 0.001) loss [TFD.unVariable w, TFD.unVariable b]
    replicateM_ 1000 $ do
      () <- TFD.run trainStep -- this is necessary for haskell to select the right instance of `TFD.run`
      return ()               -- alternatively, you could annotate `replicateM_` with `Int -> IO () -> IO ()`
    -- Return the learned parameters.
    TF.Scalar w' <- TFD.run (TFD.readValue w)
    TF.Scalar b' <- TFD.run (TFD.readValue b)
    return (w', b')
```

## LICENSE ##

This project is dual licensed under [Apache 2.0](LICENSE.Apache.txt) and
[BSD3](LICENSE.BSD.txt).
