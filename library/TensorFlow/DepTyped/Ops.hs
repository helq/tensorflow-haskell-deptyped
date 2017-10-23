{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module TensorFlow.DepTyped.Ops (
  constant,
  placeholder,
  add,
  matMul
) where

import           GHC.TypeLits (Nat, Symbol)

import           Data.Vector.Sized (Vector, toList)
import           Data.Int (Int64, Int8, Int16)
import           Data.Word (Word8)
import           Data.ByteString (ByteString)

import           TensorFlow.Core (Build, MonadBuild)
import qualified TensorFlow.Ops as TF (constant, add, matMul, placeholder)
import qualified TensorFlow.Types as TF (TensorType, type (/=), Shape(Shape))
import           Data.Proxy (Proxy(Proxy))

import           TensorFlow.DepTyped.Base (ShapeProduct)
import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder, KnownNatList(natListVal), UnionPlaceholder)

constant :: forall a (s :: [Nat]) (n :: Nat).
            (TF.TensorType a, ShapeProduct s ~ n, KnownNatList s)
         => Vector n a
         -> Tensor s '[] Build a
constant v = Tensor $ TF.constant shape (toList v)
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy s)

placeholder :: forall a (shape :: [Nat]) (name :: Symbol) (n :: Nat) m.
               (TF.TensorType a, ShapeProduct shape ~ n, KnownNatList shape, MonadBuild m)
            => m (Placeholder name shape a)
placeholder = Tensor <$> TF.placeholder shape
  where shape = TF.Shape . fmap fromInteger $ natListVal (Proxy :: Proxy shape)

add :: forall a (s :: [Nat]) (p :: [(Symbol, [Nat])]) (q :: [(Symbol, [Nat])]) v1 v2.
       a TF./= Bool => Tensor s p v1 a -> Tensor s q v2 a -> Tensor s (UnionPlaceholder p q) Build a
add (Tensor t1) (Tensor t2) = Tensor (t1 `TF.add` t2)

matMul :: (TF.TensorType a, a TF./= Bool, a TF./= Int8, a TF./= Int16, a TF./= Int64, a TF./= Word8, a TF./= ByteString)
       => Tensor '[i,n] p Build a -> Tensor '[n,o] q Build a -> Tensor '[i,o] (UnionPlaceholder p q) Build a
matMul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.matMul` t2)
