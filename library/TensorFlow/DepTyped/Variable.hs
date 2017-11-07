{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.DepTyped.Variable (
  Variable(Variable, unVariable),
  initializedVariable,
  zeroInitializedVariable,
  readValue
) where

import           GHC.TypeLits (Nat)

import qualified TensorFlow.Variable as TF (Variable, initializedVariable, zeroInitializedVariable,
                                            readValue)
import qualified TensorFlow.Types as TF (TensorType, Shape(Shape))
import qualified TensorFlow.Build as TF (MonadBuild)
import           TensorFlow.Build (Build)

import           TensorFlow.DepTyped.Tensor (Tensor(Tensor))
import           TensorFlow.DepTyped.Base (KnownNatList(natListVal))
import           Data.Proxy (Proxy(Proxy))

newtype Variable (shape :: [Nat]) a = Variable { unVariable :: TF.Variable a }

initializedVariable :: forall (shape::[Nat]) a v m.
                       (TF.TensorType a, TF.MonadBuild m)
                    => Tensor shape '[] v a -> m (Variable shape a)
initializedVariable (Tensor t) = Variable <$> TF.initializedVariable t

zeroInitializedVariable :: forall (shape :: [Nat]) a m.
                           (TF.MonadBuild m, TF.TensorType a, Num a, KnownNatList shape) => m (Variable shape a)
zeroInitializedVariable = Variable <$> TF.zeroInitializedVariable shape
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)

readValue :: TF.TensorType a => Variable shape a -> Tensor shape '[] Build a
readValue (Variable v) = Tensor $ TF.readValue v
