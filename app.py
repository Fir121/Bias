from nicegui import ui
import pandas as pd
from bias_modules.llm_calls import ModelHandler, Constants
import asyncio
from bias_modules.stat import class_balance_checker, chi_square_test, text_length_classifier_deviation

# GLOBALS
df = text_col = label_col = ddesc = None

def header_bar():
    with ui.header().classes('items-baseline justify-between'):
        ui.label('DetBias').classes('text-3xl')
        ui.label('A bias detection and mitigation tool by developers, for developers').classes('text-md')


async def stage1():
    global df, text_col, label_col, ddesc

    with ui.stepper().props('vertical').classes('w-full') as stepper:
        with ui.step('Dataset Upload'):
            ui.label('Upload your training dataset (.csv)')

            def handle_file_upload(f):
                global df
                ui.notify(f'Uploaded {f.name}')
                nb1.enable()
                inp.disable()
                with open('temp.csv', 'wb') as file:
                    file.write(f.content.read())
                df = pd.read_csv('temp.csv')
                colselect.refresh()
            
            def handle_file_reject():
                ui.notify('Rejected!')
                nb1.disable()
                inp.enable()

            def clear():
                inp.reset()
                nb1.disable()
                inp.enable()
            
            inp = ui.upload(
                on_upload=handle_file_upload,
                on_rejected=handle_file_reject,
                max_file_size=1_000_000_000,
                auto_upload=True,
                multiple=False,
                max_files=1,
            ).classes('max-w-full').props("accept=.csv")

            ui.button('Clear Files', on_click=clear).classes('bg-red-500 text-white').props('flat')

            with ui.stepper_navigation():
                nb1 = ui.button('Next', on_click=stepper.next)
                nb1.disable()
            
        with ui.step('Dataset Columns'):
            @ui.refreshable
            def colselect():
                global text_col, label_col

                if df is not None:
                    col_list = df.columns.tolist()

                    def check_vals():
                        global text_col, label_col
                        if text_col.value and label_col.value:
                            nb2.enable()
                        else:
                            nb2.disable()

                    ui.label('Select the column containing the text')
                    text_col = ui.select(options=col_list, on_change=check_vals)

                    ui.label('Select the label column')
                    label_col = ui.select(options=col_list, on_change=check_vals)

                    def move_next():
                        stepper.next()
                        desc_entry.refresh()

                    with ui.stepper_navigation():
                        nb2 = ui.button('Next', on_click=move_next)
                        nb2.disable()
                        ui.button('Back', on_click=stepper.previous).props('flat')
                else:
                    ui.label('Upload a dataset first!')
            colselect()

        with ui.step('Dataset Description'):
            @ui.refreshable
            async def desc_entry():
                global text_col, label_col, ddesc

                loop = asyncio.get_running_loop()
                temp = ui.spinner(size='lg')

                if not ddesc and text_col and label_col and text_col.value and label_col.value:
                    llm = ModelHandler()
                    prompt = Constants.get_classifier_determination_system_prompt(label_col.value, text_col.value, df[label_col.value].unique().tolist())
                    ddesc = await loop.run_in_executor(None, llm.generate_response, prompt)

                def set_val(val):
                    global ddesc
                    ddesc = val.value

                temp.delete()

                ui.label('What is the purpose of this dataset?')
                ta = ui.textarea(value=ddesc, on_change=set_val).classes('w-1/2')
                with ui.stepper_navigation():
                    ui.button('Process Bias', on_click=lambda: ui.notify(ddesc, type='positive')).bind_enabled_from(ta, 'value')
                    ui.button('Back', on_click=stepper.previous).props('flat')
            
            await desc_entry()

@ui.page('/', dark=True)
async def index():
    header_bar()

    ui.label('Select your current stage of work, for further analysis:').classes('text-sm')
    with ui.tabs() as tabs:
        ui.tab('Data Stage')
        ui.tab('Model Testing')

    with ui.tab_panels(tabs, value='Data Stage').classes('w-full'):
        with ui.tab_panel('Data Stage').classes('w-full'):
            await stage1()

        with ui.tab_panel('Model Testing').classes('w-full'):
            ui.label('Content B')

ui.run()