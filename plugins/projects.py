from __future__ import unicode_literals
from nikola.plugin_categories import Task
from nikola import utils

import os


class Projects(Task):
    """Render project indexes."""

    name = 'projects'
    dates = {}
    conf_project_path = 'projects'

    def set_site(self, site):
        """Set Nikola site."""
        site.register_path_handler('project', self.project_path)
        self.conf_project_path = site.config.get('PROJECT_PATH', 'projects')
        site._GLOBAL_CONTEXT['project_path'] = self.conf_project_path
        site._GLOBAL_CONTEXT['project_index'] = {}
        for lang, tpath in site.config['TRANSLATIONS'].items():
            site._GLOBAL_CONTEXT['project_index'][lang] = '/' + os.path.join(tpath, self.conf_project_path, site.config['INDEX_FILE']).replace('\\', '/')

        return super(Projects, self).set_site(site)

    def project_path(self, name, lang):
        """Generate links to project pages."""
        return [_f for _f in self.projects[name].permalink(lang).split('/') if _f]

    def is_project(self, p):
        """Test projecthood of a page."""
        return p.destination_path(lang=self.kw['default_lang']).startswith(self.conf_project_path)

    def find_projects(self):
        """Find all projects."""
        self._projects = [p for p in self.site.timeline if self.is_project(p)]
        # Remove projects from timeline so that they don't get rendered separately
        #self.site.timeline = [p for p in self.site.timeline if not self.is_project(p)]

    @property
    def projects(self):
        """Look for projects if we haven’t already."""
        try:
            return self._projects
        except AttributeError:
            self.find_projects()
            return self._projects

    def gen_tasks(self):
        """Render project list."""

        self.kw = {
            'project_path': self.conf_project_path,
            'index_file': self.site.config['INDEX_FILE'],
            'strip_indexes': self.site.config['STRIP_INDEXES'],
            'output_folder': self.site.config['OUTPUT_FOLDER'],
            'cache_folder': self.site.config['CACHE_FOLDER'],
            'default_lang': self.site.config['DEFAULT_LANG'],
            'filters': self.site.config['FILTERS'],
            'translations': self.site.config['TRANSLATIONS'],
            'global_context': self.site.GLOBAL_CONTEXT,
            'tzinfo': self.site.tzinfo,
        }

        #for k, v in self.site.GLOBAL_CONTEXT['template_hooks'].items():
        #    self.kw['||template_hooks|{0}||'.format(k)] = v._items

        yield self.group_task()

        template_name = "projects.tmpl"

        self.site.scan_posts()
        self.find_projects()

        # Create index.html for each language
        for lang in self.kw['translations']:
            # save navigation links as dependencies
            self.kw['navigation_links|{0}'.format(lang)] = self.kw['global_context']['navigation_links'](lang)

            short_tdst = os.path.join(self.kw['translations'][lang], self.kw['project_path'], self.kw['index_file'])
            tdst = os.path.normpath(os.path.join(self.kw['output_folder'], short_tdst))

            context = {}
            context["lang"] = lang
            context["projects"] = self.projects
            context["title"] = "Projects"
            context["description"] = None

            link = short_tdst.replace('\\', '/')
            index_len = len(self.kw['index_file'])
            if self.kw['strip_indexes'] and link[-(1 + index_len):] == '/' + self.kw['index_file']:
                link = link[:-index_len]
            context["permalink"] = '/' + link
            context["pagekind"] = ['projectpages']


            #TODO: I think Nikola sorts the posts chronologically
            # when scan_posts() is called.

            template_dep = self.site.template_system.template_deps(template_name)
            file_dep = []

            for p in self.projects:
                file_dep += [p.translated_base_path(l) for l in self.kw['translations'] if l in p.translated_to]
            

            yield utils.apply_filters({
                'basename': self.name,
                'name': tdst,
                'file_dep': file_dep + template_dep,
                'targets': [tdst],
                'actions': [
                    (self.site.render_template, (template_name, tdst, context))],
                'clean': True,
                'uptodate': [utils.config_changed(self.kw, 'projects:html:' + tdst)],
            }, self.kw['filters'])

