
#ifndef Kuiper_KB_CrossEntropyLoss_H
#define Kuiper_KB_CrossEntropyLoss_H

#include <kuiper.h>
#include <kbench.h>

float Kuiper_KB_CrossEntropyLoss_ce_loss_fw_f32(uint32_t b, uint32_t c,
                                                float *predictions,
                                                uint32_t *targets);

#define Kuiper_KB_CrossEntropyLoss_H_DEFINED
#endif /* Kuiper_KB_CrossEntropyLoss_H */
