{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module TensorFlow.DepTyped.Base (
  KnownNatList(natListVal),
  ShapeProduct
) where

import           GHC.TypeLits (Nat, KnownNat, natVal, type (*))
import           Data.Proxy (Proxy(Proxy))

class KnownNatList (ns :: [Nat]) where
   natListVal :: proxy ns -> [Integer]
-- Base case
instance KnownNatList '[] where
  natListVal _ = []
-- Inductive step
instance (KnownNat n, KnownNatList ns) => KnownNatList (n ': ns) where
  natListVal _ = natVal (Proxy :: Proxy n) : natListVal (Proxy :: Proxy ns)

type family ShapeProduct (s :: [Nat]) :: Nat where
  ShapeProduct '[] = 1
  ShapeProduct (m ': s) = m * ShapeProduct s
