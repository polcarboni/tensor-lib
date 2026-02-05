#pragma once

namespace tensor
{
    // -------------------------------------------------------------------------------------------------------------  
    //                                           OPERATOR DEFINITIONS (Functors)
    // -------------------------------------------------------------------------------------------------------------     
    
    
    // ------------------------------------------------ unary operators --------------------------------------------

    template <typename T>
    struct AbsFunctor {
        inline T operator()(T a) const
        {
            auto x = T();
            return x;
        }
    };

    template <typename T>
    struct ExpFunctor {
        inline T operator()(T a) const
        {

        }
    };

    template <typename T>
    struct ReLUFunctor {
        inline T operator()(T a) const
        {
            // max(0, a)
        } 
    };


    // ------------------------------------------------ binary operators --------------------------------------------     

    template <typename T>
    struct AddFunctor {
        inline T operator()(T a, T b) const
        {

        }
    };

    template <typename T>
    struct MulFunctor {
        inline T operator()(T a, T b) const
        {

        }
    };

} // namespace tensor