{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE UndecidableInstances #-}

module TensorFlow.DepTyped.Ops (
  constant,
  placeholder,
  add,
  matMul,
  argMax,
  softmax,
  scalar,
  oneHot,
  oneHot_
) where

import           GHC.TypeLits (Nat, Symbol, KnownNat, natVal, TypeError, ErrorMessage(Text, ShowType, (:<>:)), type (-))
import           Data.Proxy (Proxy(Proxy))
import           Data.Promotion.Prelude (type Length)

import           Data.Vector.Sized (Vector, toList)
import           Data.Int (Int8, Int16, Int32, Int64)
import           Data.Word (Word8, Word16)
import           Data.Complex (Complex)

import           TensorFlow.Core (Build, MonadBuild)
import qualified TensorFlow.Ops as TF (constant, add, matMul, placeholder, argMax, scalar, softmax, oneHot)
import qualified TensorFlow.Types as TF (TensorType, type (/=), Shape(Shape), type OneOf)

import           TensorFlow.DepTyped.Base (KnownNatList(natListVal), ShapeProduct)
import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder, UnionPlaceholder)

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

matMul :: (TF.OneOf '[(Complex Double), (Complex Float), Int32, Word16, Double, Float] a)
       => Tensor '[i,n] p Build a -> Tensor '[n,o] q Build a -> Tensor '[i,o] (UnionPlaceholder p q) Build a
matMul (Tensor t1) (Tensor t2) = Tensor (t1 `TF.matMul` t2)


type family RemoveAxisFromShape (idx::Nat) (shape::[Nat]) :: [Nat] where
  RemoveAxisFromShape idx shape = RemoveAxisFromShape' idx shape idx shape

type family RemoveAxisFromShape' (idx::Nat) (shape::[Nat]) (idxorig::Nat) (shapeorig::[Nat]) :: [Nat] where
  RemoveAxisFromShape' _ '[]     idx shape = TypeError ('Text "Index " ':<>: 'ShowType idx ':<>:
                                                        'Text " is out of bounds of shape " ':<>: 'ShowType shape ':<>:
                                                        'Text ". Valid values for index [" ':<>:
                                                        'ShowType 0 ':<>: 'Text ".." ':<>: 'ShowType (Length shape) ':<>: 'Text "]" )
  RemoveAxisFromShape' 0 (_:shs) _ _ = shs
  RemoveAxisFromShape' n (sh:shs) idx shape = sh : RemoveAxisFromShape' (n-1) shs idx shape

argMax :: (KnownNat n,
           output_shape ~ RemoveAxisFromShape n shape,
           TF.OneOf '[(Complex Double), (Complex Float), Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] t,
           TF.OneOf '[Int32, Int64] output_type)
       => Tensor shape phs v t
       -> Proxy n
       -> Tensor output_shape phs Build output_type
argMax (Tensor t) p = Tensor $ TF.argMax t (TF.scalar (fromInteger $ natVal p :: Int64))

softmax :: (TF.OneOf '[Word16, Double, Float] t, KnownNat batchSize, KnownNat outs)
        => Tensor '[batchSize, outs] phs v t
        -> Tensor '[batchSize, outs] phs Build t
softmax (Tensor t) = Tensor $ TF.softmax t

scalar :: TF.TensorType a => a -> Tensor '[1] '[] Build a
scalar = Tensor . TF.scalar

type family AddAxisToEndShape (shape::[Nat]) (axisSize::Nat) where
  AddAxisToEndShape '[] axis = '[axis]
  AddAxisToEndShape (s:shs) axis = s : AddAxisToEndShape shs axis

oneHot :: (TF.TensorType t,
           KnownNat n,
           TF.OneOf '[Int32, Int64, Word8] tI,
           output_shape ~ AddAxisToEndShape shape n)
       => Proxy n
       -> Tensor '[1] '[] v'2 t
       -> Tensor '[1] '[] v'3 t
       -> Tensor shape phs v'1 tI
       -> Tensor output_shape phs Build t
oneHot p (Tensor t1) (Tensor t2) (Tensor tinput) = Tensor $ TF.oneHot tinput (TF.scalar (fromInteger $ natVal p :: Int32)) t1 t2

oneHot_ :: (TF.TensorType t,
           KnownNat n,
           TF.OneOf '[Int32, Int64, Word8] tI,
           output_shape ~ AddAxisToEndShape shape n)
        => Proxy n
        -> t
        -> t
        -> Tensor shape phs v'1 tI
        -> Tensor output_shape phs Build t
oneHot_ p t1 t2 = oneHot p (scalar t1) (scalar t2)
