from jinja2 import Template
import pdb
# EVAL_PROMPT = EVAL_PROMPT.replace("\r\n", "\n")
# PROMPT_VER5_WITH_EXAMPLE = PROMPT_VER5_WITH_EXAMPLE.replace("\r\n", "\n")
# PROMPT_VER5 = PROMPT_VER5.replace("\r\n", "\n")
class prompt_build:
    @staticmethod
    def build_prompt(input_content, mode, iter_num, exm_temp = ''):
        if mode == 'eval_with_3rank':  
            prompt = EVAL_PROMPT
            prompt = prompt.format(input_content, )
        elif mode == 'generate':
            prompt = INSTRUCTION_PICK_UP.format(input_content, exm_temp)
        elif mode == 'dialogue_generate':
            prompt = DIALOGUE_GENERATE.format(input_content)
            # if iter_num > 0 :
            #     prompt = DIALOGUE_GENERATE_W_EXAMPLE
            #     prompt = prompt.format(input_content, exm_temp)
            # else:
            #     prompt = DIALOGUE_GENERATE
            #     prompt = prompt.format(input_content)
        elif mode == 'qa_generate_sep':
            prompt = QA_PROMPT_SEP.format(input_content)
        elif mode == 'get_sep_answer':
            prompt = SEP_ANSWER.format(input_content)
        elif mode == 'get_comprehensive_answer':
            prompt = COMPREHENSIVE_ANSWER.format(input_content)
        elif mode == 'multi_selection_qa_generate':
            prompt = MULTIPLE_SELECTIONS_QA.format(input_content)
        elif mode == 'pre_background_extract':
            prompt = PRE_BACKGROUND.format(input_content)
        elif mode == 'context_extract':
            prompt = POST_BACKGROUND.format(input_content, exm_temp)
        elif mode == 'answer_attribute_extract':
            prompt = ANSWER_ATTRIBUTE_EXTRACT.format(input_content)
        elif mode == 'get_candidate_w_attribute':
            prompt = SEP_ANSWER_W_ATTRIBUTE.format(input_content, exm_temp)
        elif mode == 'get_candidate':
            prompt = SEP_ANSWER.format(input_content)
        return prompt