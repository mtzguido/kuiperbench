module Kuiper.KB.Compat.Map

open FStar.Seq

(* Pure helper removed from Kuiper.Kernel.Map.  Keep it in the KuiperBench
   namespace so this project can build against an unmodified Kuiper package. *)
let lseq_map2
  (#a #b #c : Type) (#len : nat)
  (f : a -> b -> c)
  (s1 : lseq a len) (s2 : lseq b len)
  : GTot (lseq c len)
  = Seq.init_ghost len (fun i -> f (Seq.index s1 i) (Seq.index s2 i))

inline_for_extraction let () = ()
