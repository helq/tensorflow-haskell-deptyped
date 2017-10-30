{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}

module TensorFlow.DepTyped.Output (
  ControlNode(ControlNode, unControlNode)
) where

import qualified TensorFlow.Output as TF (ControlNode)
import GHC.TypeLits (Nat, Symbol)

newtype ControlNode (phs :: [(Symbol, [Nat])]) = ControlNode { unControlNode :: TF.ControlNode }
