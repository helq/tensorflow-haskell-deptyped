{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds      #-}
{-# LANGUAGE TypeInType     #-}

module TensorFlow.DepTyped.Output (
  ControlNode(ControlNode, unControlNode)
) where

import qualified TensorFlow.Output as TF (ControlNode)
import GHC.TypeLits (Nat, Symbol)
import Data.Kind (Type)

newtype ControlNode (phs :: [(Symbol, [Nat], Type)]) = ControlNode { unControlNode :: TF.ControlNode }
