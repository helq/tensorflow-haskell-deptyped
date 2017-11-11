{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE TypeInType          #-}

module TensorFlow.DepTyped.Types (
  TensorData(TensorData, unTensorData),
  encodeTensorData,
) where

import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, fromSized)

import           GHC.TypeLits (Nat, Symbol)
import           Data.Kind (Type)
import           Data.Proxy (Proxy(Proxy))

import qualified TensorFlow.Types as TF (TensorDataType, TensorData, TensorType, encodeTensorData, Shape(Shape))

import           TensorFlow.DepTyped.Base (KnownNatList(natListVal), ShapeProduct)

newtype TensorData (n :: Symbol) (s :: [Nat]) (a :: Type) = TensorData {unTensorData :: TF.TensorData a}

encodeTensorData :: forall a (name :: Symbol) (shape :: [Nat]) (n :: Nat).
                 (TF.TensorType a, ShapeProduct shape ~ n, KnownNatList shape, TF.TensorDataType VN.Vector a)
                 => Vector n a
                 -> TensorData name shape a
encodeTensorData v = TensorData (TF.encodeTensorData shape $ fromSized v :: TF.TensorData a)
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)
