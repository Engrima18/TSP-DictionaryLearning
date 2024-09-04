$\min$ & $1500w_1$   & + $1500w_2$ & + $1500w_3$ & + $1500w_4$ & + $1600h_1$ & + $1600h_2$ & + $1600h_3$ & + $1600h_4$ & \\
       & + $2000f_1$ & + $2000f_2$ & + $2000f_3$ & + $2000f_4$ & + $13o_1$   & + $13o_2$   & + $13o_3$   & + $13o_4$   & \\
       & + $15x_1$   & + $15x_2$   & + $15x_3$   & + $15x_4$   & + $3s_1$    & + $3s_2$    & + $3s_3$    &             & \\ 
\text{s.t.} & $s_0$ & = & $500$ \\ 
& $w_0$ & = & $100$ \\ 
       & $x_1$       &             &             &             &- $s_1$      &             &             &             & = $2500$ \\ 
       &             & $x_2$       &             &             &+ $s_1$      &-$s_2$       &             &             & = $5000$ \\ 
       &             &             &$x_3$        &             &             &+$s_2$       &-$s_3$       &             & = $2000$ \\ 
       &             &             &             & $x_4$       &             &             &+$s_3$       &             & = $1000$ \\ 
       & $w_1+f_1$   &             &             &             &             &             &             &             & = $100$ \\
       & $-w_1$      & + $w_2+f_2$ &             &             &             & -$h_2$      &             &             & = $0$ \\
       &             &$-w_2$       &  + $w_3+f_3$&             &             &             &-$h_3$       &             & = $0$ \\
       &             &             & $-w_3$      & + $w_4+f_4$ &             &             &             & -$h_4$      & = $0$ \\
       & $160w_1- 4x_1$            &             &             &+$o_1$       &             &             &             & $\geq 0$ \\
       &             &$160w_2- 4x_2$ &           &             &             &+$o_2$       &             &             & $\geq 0$ \\
       &             &             &$160w_3- 4x_3$&            &             &             &+$o_3$       &             & $\geq 0$ \\
       &             &             &             &$160w_4- 4x_4$&            &             &             &+$o_4$       & $\geq 0$ \\
       & $20w_1$     &             &             &             &+$o_1$       &             &             &+$o_4$       & $\geq 0$ \\
       &             &$20w_2$      &             &             &             &-$o_2$       &             &             & $\geq 0$ \\
       &             &             &$20w_3$      &             &             &             &-$o_3$       &             & $\geq 0$ \\
       &             &             &             &$20w_4$      &             &             &             &-$o_4$       & $\geq 0$ \\ 
       & $w_1$,      & $w_2$,      & $w_3$,      & $w_4$,      & $h_1$,      & $h_2$,      & $h_3$,      & $h_4$       & $\geq 0$ \\ 
       & $f_1$,      & $f_2$,      & $f_3$,      & $f_4$,      & $o_1$,      & $o_2$,      & $o_3$,      & $o_4$       & $\geq 0$ \\
       & $x_1$,      & $x_2$,      & $x_3$,      & $x_4$,      & $s_1$,      & $s_2$,      & $s_3$,      & $s_4$       & $\geq 0$ \\ 
       & $x_1$,      & $x_2$,      & $x_3$,      & $x_4$,      & $s_1$,      & $s_2$,      & $s_3$,      & $s_4$       & $\in \mathbb{Z}_+$ 


    P_A &= 8 \\
    P_B &= 70 \\
    P_C &= 100 \\
    H_A &= 1 \\
    H_B &= 2 \\
    H_C &= 3 \\
    U_B &= 2 \\
    U_C &= 1 \\

    P_A &= 8  \quad \text{(prezzo unitario del farmaco A)} \\
    P_B &= 70 \quad \text{(prezzo unitario del farmaco B)} \\
    P_C &= 100\quad \text{(prezzo unitario del farmaco C)} \\
    H_A &= 1  \quad \text{(ore di manodopera per produrre una unità di A)} \\
    H_B &= 2  \quad \text{(ore di manodopera per produrre una unità di B)} \\
    H_C &= 3  \quad \text{(ore di manodopera per produrre una unità di C)} \\
    U_B &= 2  \quad \text{(unità di A richieste per produrre una unità di B)} \\
    U_C &= 1  \quad \text{(unità di B richieste per produrre una unità di C)} \\