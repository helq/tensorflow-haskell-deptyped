{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module TensorFlow.DepTyped.Types (
  TensorData(TensorData),
  encodeTensorData
) where

import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, fromSized)

import           GHC.TypeLits (Nat, Symbol)
import           Data.Proxy (Proxy(Proxy))

import qualified TensorFlow.Types as TF (TensorDataType, TensorData, TensorType, encodeTensorData, Shape(Shape))

import           TensorFlow.DepTyped.Base (KnownNatList(natListVal), ShapeProduct)

data TensorData (n :: Symbol) (s :: [Nat]) a where
  TensorData :: (TF.TensorType a) => TF.TensorData a -> TensorData n s a

encodeTensorData :: forall a (name :: Symbol) (shape :: [Nat]) (n :: Nat).
                 (TF.TensorType a, ShapeProduct shape ~ n, KnownNatList shape, TF.TensorDataType VN.Vector a)
                 => Vector n a
                 -> TensorData name shape a
encodeTensorData v = TensorData (TF.encodeTensorData shape $ fromSized v :: TF.TensorData a)
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)
